"""Local-only, prespecified seed sensitivity. Export aggregates; never patient rows."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
import traceback
import warnings

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
START = time.monotonic()
STAGE = "initialization"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def progress(stage, **public):
    global STAGE
    STAGE = stage
    write("progress.json", {"stage": stage, "seconds": round(time.monotonic()-START, 1), **public})


def initialize():
    sys.path.insert(0, str(SOURCE / "src"))
    import numpy as np
    import pandas as pd
    import torch
    from ctb.models.fold_preprocessing import FoldPreprocessor
    from ctb.models.locked_validation import make_grouped_splits, _seed
    from ctb.models.archetype_contextualized import LinearArchetypeContextualized, TrainConfig, train
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return np, pd, torch, FoldPreprocessor, make_grouped_splits, _seed, LinearArchetypeContextualized, TrainConfig, train


def synthetic_checks():
    np, pd, torch, Pre, splits, seed, Model, Config, train = initialize()
    # Synthetic only: repeated groups cannot enter both sides, every row held out once.
    y = np.tile([0, 1], 100)
    groups = np.repeat(np.arange(100), 2)
    for a, b in splits(y, groups, n_splits=5, seed=42, label="synthetic"):
        assert not set(groups[a]).intersection(groups[b])
    c = pd.DataFrame({"numeric": [0., 2., np.nan, 4.], "category": ["a", "b", "a", "b"], "empty": [np.nan]*4})
    x = pd.Series([{"A"}, {"A"}, {"B"}, {"A"}])
    pre = Pre(("numeric", "empty"), ("category",), min_treatment_prevalence=.5)
    training = pre.fit_transform(c, x)
    before = (pre.numeric_medians_.copy(), pre.numeric_means_.copy(), pre.categorical_vocabulary_.copy(), pre.treatment_vocabulary_)
    test_c = pd.DataFrame({"numeric": [1000000.], "category": ["heldout"], "empty": [1.]})
    evaluation = pre.transform(test_c, pd.Series([{"HELDOUT"}]))
    assert before == (pre.numeric_medians_, pre.numeric_means_, pre.categorical_vocabulary_, pre.treatment_vocabulary_)
    assert "heldout" not in pre.categorical_vocabulary_["category"]
    assert "HELDOUT" not in pre.treatment_vocabulary_
    assert "empty" not in pre.retained_numeric_columns_
    assert evaluation.C.shape[1] == training.C.shape[1]
    predictions = []
    for _ in range(2):
        s = seed(42, 0, 4)
        model = Model(training.C.shape[1], training.X.shape[1], 5, simplex=True, seed=s)
        train(model, training.C, training.X, np.array([0., 1., 0., 1.]), Config(epochs=5), seed=s)
        predictions.append(model.predict_proba(training.C, training.X))
    assert np.array_equal(*predictions)
    from sklearn.metrics import roc_auc_score
    toy_y, toy_p = np.array([0, 1, 0, 1]), np.array([.2, .3, .5, .9])
    selected = np.array([0, 0, 1, 2, 3, 3])
    weights = np.bincount(selected, minlength=4)
    assert roc_auc_score(toy_y[selected], toy_p[selected]) == roc_auc_score(toy_y, toy_p, sample_weight=weights)
    return {"synthetic_only": True, "group_disjointness": True, "train_only_unknown_features": True, "seed_before_initialization_repeat_identical": True, "weighted_cluster_bootstrap_matches_explicit_resampling": True}


def main(args):
    protocol = json.loads((ROOT / "protocol_before_run.json").read_text())
    assert digest(Path(__file__)) == protocol["driver_sha256"]
    for path, sha in protocol["source_sha256"].items():
        assert digest(SOURCE / path) == sha
    np, pd, torch, Pre, make_splits, seed_rule, Model, Config, train = initialize()
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
    from ctb.data.clean import clean_patient_cases, clean_regimens, build_cohort
    from ctb.data.load import load_patient_cases, load_regimens, load_biochemistry, load_dst
    from ctb.data.conversion import derive_conversion
    from ctb.data.context import build_context_matrix
    from ctb.data.treatments import select_initial_regimens
    from ctb.models.locked_validation import LockedValidationConfig, RawValidationData, build_schema6_raw_validation_data, resolve_patient_groups

    progress("verify_governed_inputs")
    for name, sha in protocol["raw_sha256"].items():
        assert digest(args.raw_dir / name) == sha
    write("synthetic_validation.json", synthetic_checks())
    progress("build_endpoint_specific_cohorts")
    cases = clean_patient_cases(load_patient_cases(args.raw_dir))
    regimens = clean_regimens(load_regimens(args.raw_dir))
    bio = load_biochemistry(args.raw_dir)
    conversion = derive_conversion(load_dst(args.raw_dir))
    consort = build_cohort(cases, regimens)
    initial = select_initial_regimens(regimens)
    cfg = LockedValidationConfig()
    datasets = {"death_failure": build_schema6_raw_validation_data(cases, regimens, bio, cfg)}
    baseline_positive = set(conversion.index[conversion["first_pos_day"].le(0)])
    for endpoint, outcome_column in (("observed_conversion", "event"), ("day60_negative", "day60_status")):
        eligible = [cid for cid in consort.condition_ids if cid in baseline_positive and pd.notna(conversion.at[cid, outcome_column])]
        idx = pd.Index(eligible, name="condition_id")
        treatment = initial.reindex(idx)
        assert treatment["initial_start"].notna().all()
        assert treatment["drug_set"].map(len).gt(0).all()
        context, spec = build_context_matrix(cases, bio, eligible, baseline_index_dates=treatment["initial_start"], preserve_degenerate_columns=True)
        datasets[endpoint] = RawValidationData(context=context.reindex(idx), treatments=pd.Series(treatment["drug_set"].to_numpy(), index=idx), outcome=conversion[outcome_column].reindex(idx).astype(int), patient_groups=resolve_patient_groups(cases, idx), numeric_columns=tuple(spec.numeric), categorical_columns=tuple(spec.categorical))
    results = {}
    seeds = protocol["repeated_cv_seeds"]
    metric_names = ("auroc", "auprc", "brier", "log_loss")

    def metrics(y, p):
        assert p.shape == y.shape and np.isfinite(p).all()
        return dict(zip(metric_names, (float(roc_auc_score(y, p)), float(average_precision_score(y, p)), float(brier_score_loss(y, p)), float(log_loss(y, p)))))

    for endpoint, data in datasets.items():
        y = np.asarray(data.outcome, dtype=np.int64)
        groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
        if min(int(y.sum()), int((1-y).sum()), len(np.unique(groups))) < 30:
            results[endpoint] = {"status": "not_evaluable", "reason": "endpoint does not meet prespecified disclosure/modelability gate", "counts_suppressed": True}
            write(f"{endpoint}_report.json", results[endpoint])
            continue
        configurations = [("shared_patch", 500, 1.)]
        if endpoint == "observed_conversion":
            configurations.append(("historical_recipe_sensitivity", 2000, 0.))
        names = ["c_only", "x_only", "concatenated", *[c[0] for c in configurations]]
        repeated = {name: [] for name in names}
        records = []
        fixed_fold_initializations = []
        for run_seed in seeds:
            splits = make_splits(y, groups, n_splits=5, seed=run_seed, label=endpoint)
            split_hash = hashlib.sha256()
            for tr, te in splits:
                split_hash.update(np.asarray(tr, dtype="<i8").tobytes())
                split_hash.update(np.asarray(te, dtype="<i8").tobytes())
                # Fold class counts remain private; splitter already requires both classes.
                assert len(np.unique(y[te])) == 2
            predictions = {name: np.full(len(y), np.nan) for name in names}
            additional = {s: np.full(len(y), np.nan) for s in seeds if s != 42} if endpoint == "death_failure" and run_seed == 42 else {}
            for fold, (tr, te) in enumerate(splits):
                progress("fit", endpoint=endpoint, split_seed=run_seed, fold=fold+1, folds=5)
                pre = Pre(data.numeric_columns, data.categorical_columns, min_treatment_prevalence=.01)
                training = pre.fit_transform(data.context.iloc[tr], data.treatments.iloc[tr])
                evaluation = pre.transform(data.context.iloc[te], data.treatments.iloc[te])
                for name, tc, vc in (("c_only", training.C, evaluation.C), ("x_only", training.X, evaluation.X), ("concatenated", np.hstack([training.C, training.X]), np.hstack([evaluation.C, evaluation.X]))):
                    model = LogisticRegression(C=.1, penalty="l2", solver="lbfgs", max_iter=2000, random_state=seed_rule(run_seed, fold, 29)).fit(tc, y[tr])
                    assert (model.n_iter_ < 2000).all()
                    predictions[name][te] = model.predict_proba(vc)[:, 1]
                def shared_fit(initialization_seed, updates, entropy):
                    s = seed_rule(initialization_seed, fold, 4)
                    random.seed(s)
                    np.random.seed(s)
                    model = Model(training.C.shape[1], training.X.shape[1], 5, simplex=True, seed=s)
                    train(model, training.C, training.X, y[tr].astype(np.float64), Config(epochs=updates, lr=.01, weight_decay=.001, archetype_l1=.0005, entropy_weight=entropy), seed=s)
                    return model.predict_proba(evaluation.C, evaluation.X)
                for name, updates, entropy in configurations:
                    predictions[name][te] = shared_fit(run_seed, updates, entropy)
                for initialization_seed in additional:
                    additional[initialization_seed][te] = shared_fit(initialization_seed, 500, 1.)
            record = {"split_seed": run_seed, "initialization_seed": run_seed, "split_sha256": split_hash.hexdigest(), "metrics": {name: metrics(y, p) for name, p in predictions.items()}}
            record["paired_auroc_difference_vs_concatenated"] = {name: point["auroc"]-record["metrics"]["concatenated"]["auroc"] for name, point in record["metrics"].items() if name != "concatenated"}
            records.append(record)
            for name in names:
                repeated[name].append(predictions[name])
            if additional:
                fixed_fold_initializations = [{"initialization_seed": 42, "metrics": metrics(y, predictions["shared_patch"])}]
                fixed_fold_initializations += [{"initialization_seed": s, "metrics": metrics(y, p)} for s, p in additional.items()]
                for record_ in fixed_fold_initializations:
                    record_["paired_auroc_difference_vs_concatenated"] = record_["metrics"]["auroc"]-record["metrics"]["concatenated"]["auroc"]
            # Check the frozen primary reference without selecting a seed or changing settings.
            if endpoint == "death_failure" and run_seed == 42:
                assert abs(record["metrics"]["shared_patch"]["auroc"]-.8070905138252038) < 1e-10
                assert abs(record["metrics"]["concatenated"]["auroc"]-.8074976205487047) < 1e-10
            # Write progress-safe aggregate partial results so completed fits remain reviewable.
            write(f"{endpoint}_seed_metrics.json", {"endpoint": endpoint, "runs_completed": len(records), "runs": records, "fixed_seed42_fold_initialization_sensitivity": fixed_fold_initializations})

        progress("patient_cluster_bootstrap", endpoint=endpoint, resamples=protocol["bootstrap"]["resamples"])
        unique, inverse = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(protocol["bootstrap"]["seed"])
        boot = {name: [] for name in names}
        for b in range(protocol["bootstrap"]["resamples"]):
            multiplicities = np.bincount(rng.integers(0, len(unique), size=len(unique)), minlength=len(unique))
            weights = multiplicities[inverse]
            if np.sum(weights[y == 0]) == 0 or np.sum(weights[y == 1]) == 0:
                continue
            for name in names:
                boot[name].append(float(np.mean([roc_auc_score(y, p, sample_weight=weights) for p in repeated[name]])))
            if (b+1) % 100 == 0:
                progress("patient_cluster_bootstrap", endpoint=endpoint, completed=b+1, resamples=protocol["bootstrap"]["resamples"])
        summarized = {}
        for name in names:
            summarized[name] = {key: {"mean": float(np.mean([r["metrics"][name][key] for r in records])), "median": float(np.median([r["metrics"][name][key] for r in records])), "range": [float(min(r["metrics"][name][key] for r in records)), float(max(r["metrics"][name][key] for r in records))]} for key in metric_names}
            summarized[name]["mean_auroc_patient_bootstrap_ci95"] = np.percentile(boot[name], [2.5, 97.5]).tolist()
        paired = {}
        for comparator in ["c_only", "x_only", "concatenated"]:
            differences = [r["metrics"]["shared_patch"]["auroc"]-r["metrics"][comparator]["auroc"] for r in records]
            paired[comparator] = {"mean_difference": float(np.mean(differences)), "range": [min(differences), max(differences)], "seeds_patch_higher": sum(d > 0 for d in differences), "ci95": np.percentile(np.asarray(boot["shared_patch"])-np.asarray(boot[comparator]), [2.5, 97.5]).tolist()}
        decision = all(point["mean_difference"] > 0 and point["ci95"][0] > 0 and point["seeds_patch_higher"] >= 4 for point in paired.values())
        results[endpoint] = {"cohort": {"conditions": len(y), "patients": len(unique), "event": int(y.sum()), "non_event": int((1-y).sum())}, "runs": records, "summary": summarized, "shared_patch_paired_auroc": paired, "prespecified_secondary_superiority_rule_satisfied": decision, "fixed_seed42_fold_initialization_sensitivity": fixed_fold_initializations, "fold_contract": {"patient_disjoint": True, "all_models_identical_folds_within_repeat": True, "all_rows_heldout_once_per_repeat": True, "train_only_preprocessing": True}, "bootstrap_valid": len(boot["shared_patch"])}
        write(f"{endpoint}_report.json", results[endpoint])
    report = {"protocol_sha256": digest(ROOT / "protocol_before_run.json"), "driver_sha256": digest(Path(__file__)), "endpoints": results, "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__, "torch_threads": 1}, "privacy": {"row_values_exported": False, "predictions_exported": False, "small_cell_threshold": 30}, "elapsed_seconds": round(time.monotonic()-START, 1)}
    write("validation_report.json", report)
    progress("complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--dua-acknowledge", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    try:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            if args.synthetic_only:
                write("synthetic_validation.json", synthetic_checks())
            else:
                assert args.dua_acknowledge and args.raw_dir
                main(args)
        print(json.dumps({"status": "complete", "synthetic_only": args.synthetic_only, "private_values_printed": False}))
    except BaseException as error:
        trace = traceback.extract_tb(error.__traceback__)
        safe = {"status": "failed", "stage": STAGE, "error_type": type(error).__name__, "code_location": f"{Path(trace[-1].filename).name}:{trace[-1].lineno}" if trace else None, "private_values_printed": False}
        write("failure_metadata.json", safe)
        print(json.dumps(safe))
        raise SystemExit(1) from None
