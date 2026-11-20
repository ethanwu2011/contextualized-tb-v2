"""Governed local rebuild for the shared-primary PATCH manuscript figures.

Private source rows, identifiers, loadings, coefficients per condition, coordinates,
and predictions remain in memory. Only disclosure-gated aggregates are written.
Requires the same approved August 2023 export hashes as the locked analyses.
"""

# ruff: noqa: E402 -- Resolve the portable governed-source root before imports.

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import sys
import warnings
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--source-root", type=Path)
_source_args, _ = _bootstrap.parse_known_args()
_packaged = next(
    (
        p
        for p in Path(__file__).resolve().parents
        if (p / "src/ctb").is_dir() and (p / "scripts/run_shared_k_sensitivity.py").is_file()
    ),
    None,
)
SOURCE = _source_args.source_root or _packaged or WORKSPACE / "contextualized-tb-v2_release_audit"
ANALYSIS_SCRIPTS = (
    SOURCE / "scripts"
    if (SOURCE / "scripts/run_shared_k_sensitivity.py").is_file()
    else WORKSPACE / "PATCH_nonnegotiable_analyses_2026-08-27/scripts"
)
FROZEN = (
    SOURCE / "manuscript/aggregate_results"
    if _packaged or _source_args.source_root
    else WORKSPACE / "PATCH_nonnegotiable_analyses_2026-08-27/aggregate_results"
)
sys.path[:0] = [str(SOURCE / "src"), str(ANALYSIS_SCRIPTS)]

import numpy as np
import pandas as pd
import torch

from ctb.data.clean import clean_patient_cases, clean_regimens
from ctb.data.load import load_biochemistry, load_patient_cases, load_regimens
from ctb.data.temporal import extract_calendar_year, patient_disjoint_temporal_masks
from ctb.models.archetype_contextualized import LinearArchetypeContextualized, train
from ctb.models.fold_preprocessing import FoldPreprocessor
from ctb.models.locked_validation import (
    LockedValidationConfig,
    _additive_probability,
    _fit_additive,
    _seed,
    build_schema6_raw_validation_data,
    make_grouped_splits,
)
from run_aggregate_sensitivities import _cluster_bootstrap_metrics, _point_metrics
from run_shared_k_sensitivity import OOF_CONFIG
from run_shared_profile_full_pipeline_stability import PUBLISHED_CONFIG
from run_shared_profile_reference_composition import _modality_of

MIN_CELL = 30
TABLE_RANKS = [1, 2, 3, 5, 4]
HASHES = {
    "TB_Portals_Patient_Cases_August_2023.csv": "8a4aacc5dc15743932d55eae72b27a41fc8404b2b4bb6bed19e3796ae33283db",
    "TB_Portals_Regimens_August_2023.csv": "9ae4554beacaa5829186b1ebcff79aa57e11cec92d81460e9a53f6178cb26699",
    "TB_Portals_Biochemistry_August_2023.csv": "fe6690ec5e1c07e0da35616dafc2e7f819267346524ec73c7c77bab9b3aacc61",
}
GOVERNANCE = {
    "row_level_objects_serialized": False,
    "identifiers_serialized": False,
    "small_cell_threshold": MIN_CELL,
    "private_coordinates_serialized": False,
}


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    report["governance"] = GOVERNANCE
    report["runtime"] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fit(matrices, y, seed, config):
    model = LinearArchetypeContextualized(
        matrices.C.shape[1],
        matrices.X.shape[1],
        n_archetypes=5,
        simplex=True,
        seed=seed,
    )
    return train(model, matrices.C, matrices.X, y.astype(np.float64), config, seed=seed)


def prepare(data, cfg, train_idx, test_idx):
    pre = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=cfg.min_treatment_prevalence,
    )
    a = pre.fit_transform(data.context.iloc[train_idx], data.treatments.iloc[train_idx])
    b = pre.transform(data.context.iloc[test_idx], data.treatments.iloc[test_idx])
    return a, b


def nb(y, p, t):
    flag = p >= t
    return float(np.sum(flag & (y == 1)) / len(y) - np.sum(flag & (y == 0)) / len(y) * t / (1 - t))


def validation(data, cfg, out):
    y = np.asarray(data.outcome, dtype=int)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    predictions = {k: np.full(len(y), np.nan) for k in ["additive", "shared_k5"]}
    splits = make_grouped_splits(
        y,
        groups,
        n_splits=cfg.outer_splits,
        seed=cfg.seed,
        label="shared K sensitivity",
    )
    for fold, (tr, te) in enumerate(splits):
        a, b = prepare(data, cfg, tr, te)
        additive = _fit_additive(
            a,
            y[tr],
            c_value=0.1,
            max_iter=cfg.logistic_max_iter,
            seed=_seed(cfg.seed, fold, 1),
        )
        predictions["additive"][te] = _additive_probability(additive, b)
        # K=5 is position=4 in the frozen K=3,...,8 sensitivity driver.
        model = fit(a, y[tr], _seed(cfg.seed, fold, 4), OOF_CONFIG)
        predictions["shared_k5"][te] = model.predict_proba(b.C, b.X)
    assert all(np.isfinite(p).all() for p in predictions.values())
    frozen = json.loads((FROZEN / "shared_k_sensitivity.json").read_text())
    metrics = {k: _point_metrics(y, p) for k, p in predictions.items()}
    assert all(
        abs(metrics[k][m] - frozen["oof_metrics"][k][m]) < 1e-6
        for k in predictions
        for m in ["auroc", "auprc", "brier", "log_loss"]
    ), "OOF aggregate mismatch"
    thresholds = np.arange(0.05, 0.401, 0.025)
    unique, inverse = np.unique(groups, return_inverse=True)
    grouped = [np.flatnonzero(inverse == i) for i in range(len(unique))]
    rng = np.random.default_rng(20261023)
    draws = np.empty((500, len(thresholds)))
    for b in range(500):
        rows = np.concatenate([grouped[i] for i in rng.integers(0, len(unique), len(unique))])
        for j, t in enumerate(thresholds):
            draws[b, j] = nb(y[rows], predictions["shared_k5"][rows], t) - nb(
                y[rows], predictions["additive"][rows], t
            )
    dca = []
    for j, t in enumerate(thresholds):
        row = {
            "threshold": float(t),
            "monitor_all_net_benefit": float(y.mean() - (1 - y.mean()) * t / (1 - t)),
            "monitor_none_net_benefit": 0.0,
            "shared_minus_additive_ci95": np.percentile(draws[:, j], [2.5, 97.5]).tolist(),
        }
        for k, p in predictions.items():
            flag = p >= t
            tp = int(np.sum(flag & (y == 1)))
            fp = int(np.sum(flag & (y == 0)))
            assert min(tp, fp, int(y.sum()) - tp, int((y == 0).sum()) - fp) >= MIN_CELL
            row[k + "_net_benefit"] = nb(y, p, t)
            row[k + "_flagged_n"] = int(flag.sum())
            row[k + "_event_per_1000"] = 1000 * tp / len(y)
            row[k + "_nonevent_per_1000"] = 1000 * fp / len(y)
        row["shared_minus_additive_net_benefit"] = (
            row["shared_k5_net_benefit"] - row["additive_net_benefit"]
        )
        dca.append(row)
    calibration = {}
    pr = {}
    for k, p in predictions.items():
        bins = []
        for indices in np.array_split(np.argsort(p, kind="mergesort"), 10):
            events = int(y[indices].sum())
            if min(events, len(indices) - events) < MIN_CELL:
                continue
            bins.append(
                {
                    "n": len(indices),
                    "mean_predicted": float(p[indices].mean()),
                    "observed_rate": float(y[indices].mean()),
                }
            )
        calibration[k] = bins
        curve = []
        for t in np.linspace(0, 1, 101):
            flag = p >= t
            tp = int(np.sum(flag & (y == 1)))
            fp = int(np.sum(flag & (y == 0)))
            fn = int(np.sum(~flag & (y == 1)))
            tn = int(np.sum(~flag & (y == 0)))
            if min(tp, fp, fn, tn) < MIN_CELL:
                continue
            curve.append(
                {
                    "threshold": float(t),
                    "precision": tp / (tp + fp),
                    "recall": tp / int(y.sum()),
                }
            )
        pr[k] = curve
    report = {
        "analysis_id": "shared_patch_k5_primary_oof_clinical_validation",
        "estimator": "shared simplex K=5; original shared architecture",
        "configuration": {
            "outer_folds": cfg.outer_splits,
            "patient_grouped": True,
            "fold_local_preprocessing": True,
            "epochs": OOF_CONFIG.epochs,
            "seed_rule": "_seed(config.seed, fold_index, 4)",
            "reference_match_tolerance": 1e-6,
        },
        "cohort": {
            "conditions": len(y),
            "patients": len(unique),
            "events": int(y.sum()),
            "observed_non_events": int(len(y) - y.sum()),
        },
        "frozen_oof_metrics_reproduced": True,
        "metrics": metrics,
        "patient_clustered_uncertainty": frozen["patient_clustered_uncertainty"],
        "calibration_bins": calibration,
        "precision_recall_curves": pr,
        "threshold_rows": dca,
        "decision_curve_bootstrap": {
            "unit": "patient",
            "resamples": 500,
            "seed": 20261023,
        },
        "interpretation": "Exploratory internal prognostic risk targeting; intervention benefit and treatment selection were not tested.",
    }
    write(out / "shared_oof_clinical_validation.json", report)


def transport(cases, data, cfg, out):
    y = np.asarray(data.outcome, dtype=int)
    groups = data.patient_groups.astype("string").str.strip().to_numpy(dtype=str)
    indexed = cases.drop_duplicates("condition_id").set_index("condition_id")
    years = extract_calendar_year(indexed["registration_date"]).reindex(data.context.index)
    tr, te, _ = patient_disjoint_temporal_masks(
        years,
        pd.Series(groups, index=data.context.index, dtype="string"),
        train_through=2020,
        test_from=2021,
    )

    def run(train_idx, test_idx, seed):
        a, b = prepare(data, cfg, train_idx, test_idx)
        additive = _fit_additive(
            a, y[train_idx], c_value=0.1, max_iter=cfg.logistic_max_iter, seed=seed
        )
        shared = fit(a, y[train_idx], _seed(seed, 31), OOF_CONFIG)
        prob = {
            "additive": _additive_probability(additive, b),
            "shared_k5": shared.predict_proba(b.C, b.X),
        }
        return {k: _point_metrics(y[test_idx], p) for k, p in prob.items()}, prob

    train_idx = np.flatnonzero(tr.to_numpy())
    test_idx = np.flatnonzero(te.to_numpy())
    assert min(np.bincount(y[test_idx], minlength=2)) >= MIN_CELL
    m, p = run(train_idx, test_idx, cfg.seed)
    temporal = {
        "definition": "Train registration year through 2020; test 2021 onward; cross-period patients excluded",
        "train_conditions": len(train_idx),
        "test_conditions": len(test_idx),
        "test_events": int(y[test_idx].sum()),
        "metrics": m,
        "patient_clustered_uncertainty": _cluster_bootstrap_metrics(
            y[test_idx],
            p,
            groups[test_idx],
            n_resamples=500,
            seed=_seed(cfg.seed, 7001),
        ),
    }
    countries = indexed["country"].astype("string").reindex(data.context.index)
    rows = []
    for name, count in sorted(countries.value_counts().items()):
        if count < 200:
            continue
        mask = countries.eq(name).fillna(False).to_numpy(dtype=bool)
        te_idx = np.flatnonzero(mask)
        if min(np.bincount(y[te_idx], minlength=2)) < MIN_CELL:
            continue
        tr_idx = np.flatnonzero(~mask & ~np.isin(groups, np.unique(groups[te_idx])))
        metrics, _ = run(tr_idx, te_idx, _seed(cfg.seed, len(rows), 8100))
        rows.append(
            {
                "held_out_country": str(name),
                "test_conditions": len(te_idx),
                "test_events": int(y[te_idx].sum()),
                "metrics": metrics,
            }
        )
    write(
        out / "shared_internal_transport.json",
        {
            "analysis_id": "shared_patch_k5_patient_disjoint_internal_transport",
            "estimator": "shared simplex K=5",
            "configuration": {
                "epochs": OOF_CONFIG.epochs,
                "preprocessing": "fit on training partition only",
                "patient_overlap": "excluded",
            },
            "temporal": temporal,
            "held_out_country": {
                "minimum_test_conditions": 200,
                "minimum_events_and_non_events": MIN_CELL,
                "rows": rows,
            },
        },
    )


def reference(cases, data, cfg, out):
    pre = FoldPreprocessor(
        data.numeric_columns,
        data.categorical_columns,
        min_treatment_prevalence=cfg.min_treatment_prevalence,
    )
    matrices = pre.fit_transform(data.context, data.treatments)
    y = np.asarray(data.outcome, dtype=int)
    model = fit(matrices, y, 0, PUBLISHED_CONFIG)
    z = model.predict_loadings(matrices.C)
    raw = model.archetypes.detach().cpu().numpy().astype(float)
    composition = json.loads((FROZEN / "shared_profile_reference_composition.json").read_text())
    hard = z.argmax(axis=1)
    assert (len(y), int(y.sum()), matrices.C.shape[1], matrices.X.shape[1]) == (
        9647,
        1770,
        279,
        26,
    )
    assert all(
        np.sum(hard == int(p["fitted_profile"]) - 1) == p["hard_count"]
        for p in composition["profiles"]
    )
    stability = json.loads((FROZEN / "shared_profile_full_pipeline_stability.json").read_text())
    drugpos = {n: j for j, n in enumerate(matrices.treatment_feature_names)}
    assert all(
        abs(
            raw[int(r["profile"]) - 1, drugpos[r["regimen_feature"]]]
            - r["reference_shared_logit_coefficient"]
        )
        < 1e-5
        for r in stability["profile_by_regimen_uncertainty"]["cells"]
        if r["support_status"] == "supported"
    )
    # Aggregate all context rows; encoded levels with either side below 30 are excluded.
    context = np.asarray(matrices.C)
    beta = z @ raw[:, : matrices.X.shape[1]]
    xc = context - context.mean(axis=0)
    sd = np.sqrt((xc**2).mean(axis=0))
    atlas = xc.T @ (beta - beta.mean(axis=0)) / len(y) / np.maximum(sd[:, None], 1e-8)
    features = []
    for j, name in enumerate(matrices.context_feature_names):
        if sd[j] < 1e-8:
            continue
        values = context[:, j]
        unique = np.unique(values)
        high = values > np.median(values) if len(unique) > 2 else values == unique.max()
        if min(int(high.sum()), int((~high).sum())) < MIN_CELL:
            continue
        numbers = []
        for k in range(matrices.X.shape[1]):
            treated = matrices.X[:, k] > 0.5
            supported = (
                min(
                    int(np.sum(high & treated)),
                    int(np.sum(high & ~treated)),
                    int(np.sum(~high & treated)),
                    int(np.sum(~high & ~treated)),
                )
                >= MIN_CELL
            )
            numbers.append(float(atlas[j, k]) if supported else None)
        if sum(v is not None for v in numbers) < 2:
            continue
        features.append(
            {
                "feature": name,
                "modality": _modality_of(name),
                "covariation": numbers,
                "rms": float(np.sqrt(np.mean(atlas[j] ** 2))),
            }
        )
    groups = list(
        dict.fromkeys(
            [
                "resistance_dst",
                "sociodemographic",
                "region",
                "bacillary_burden_clinical",
                "radiology_imaging",
                "comorbidity",
                "genomics",
                "baseline_laboratory",
                "other_clinical",
            ]
        )
    )
    selected = []
    for group in groups:
        selected.extend(
            sorted(
                [i for i, r in enumerate(features) if r["modality"] == group],
                key=lambda i: -features[i]["rms"],
            )[:6]
        )
    selected.extend(
        [
            i
            for i in sorted(range(len(features)), key=lambda i: -features[i]["rms"])
            if i not in selected
        ][: max(0, 58 - len(selected))]
    )
    selected = sorted(
        selected[:58],
        key=lambda i: (groups.index(features[i]["modality"]), -features[i]["rms"]),
    )
    for row in features:
        row.pop("rms")
    write(
        out / "shared_context_atlas.json",
        {
            "analysis_id": "shared_patch_k5_supported_context_coefficient_covariation",
            "reference_fit_reproduced": True,
            "cohort": {
                "conditions": len(y),
                "context_features": 279,
                "regimen_components": 26,
            },
            "definition": "Cov(C_j, beta_d(C))/SD(C_j), an exploratory marginal context-coefficient summary; no causal or hypothesis-test interpretation.",
            "cell_support": "At least 30 conditions in all four groups crossing above-median/other (or encoded level/other) context with regimen receipt/nonreceipt.",
            "regimen_features": list(matrices.treatment_feature_names),
            "context_features": features,
            "main_figure_feature_indices": selected,
            "full_atlas_feature_count": len(features),
            "full_reference_epochs": PUBLISHED_CONFIG.epochs,
            "reference_seed": 0,
        },
    )
    # The embedding never leaves process memory; only large supported tiles are saved.
    import umap

    embedding = umap.UMAP(
        n_components=2, random_state=42, n_neighbors=30, min_dist=0.3, n_jobs=1
    ).fit_transform(z)
    limits = np.percentile(embedding, [0.5, 99.5], axis=0)
    xe = np.linspace(limits[0, 0], limits[1, 0], 13)
    ye = np.linspace(limits[0, 1], limits[1, 1], 13)
    ix = np.digitize(embedding[:, 0], xe) - 1
    iy = np.digitize(embedding[:, 1], ye) - 1
    indexed = cases.set_index("condition_id").reindex(data.context.index)
    resistance = (
        indexed["type_of_resistance"].astype("string").fillna("Missing").to_numpy(dtype=str)
    )
    hiv = pd.to_numeric(data.context["comorb_hiv"], errors="coerce").to_numpy(dtype=float)
    smear = (
        data.context["baseline_smear"]
        .astype("string")
        .fillna("Missing")
        .str.casefold()
        .to_numpy(dtype=str)
    )
    country = data.context["country"].astype("string").fillna("Other").to_numpy(dtype=str)
    region = np.asarray(
        [v if v in {"Ukraine", "Georgia", "Nigeria", "South Africa"} else "Other" for v in country]
    )
    fitted = [
        int(next(p["fitted_profile"] for p in composition["profiles"] if p["risk_rank"] == r)) - 1
        for r in TABLE_RANKS
    ]
    rank = {profile: i for i, profile in enumerate(fitted)}
    hard = np.asarray([rank[int(v)] for v in hard])
    tiles = []
    for a in range(12):
        for b in range(12):
            mask = (ix == a) & (iy == b)
            n = int(mask.sum())
            if n < MIN_CELL:
                continue
            row = {"xbin": a, "ybin": b, "conditions": n}
            for key, values in [
                ("profile", hard),
                ("resistance", resistance),
                ("region", region),
            ]:
                levels, counts = np.unique(values[mask], return_counts=True)
                imax = int(counts.argmax())
                row[key] = None
                if int(counts[imax]) >= MIN_CELL:
                    row[key] = (
                        levels[imax].item()
                        if isinstance(levels[imax], np.generic)
                        else levels[imax]
                    )
            for key, values, known in [
                ("hiv", hiv == 1, np.isfinite(hiv)),
                ("smear", smear == "positive", smear != "missing"),
            ]:
                denom = mask & known
                pos = int(np.sum(denom & values))
                neg = int(np.sum(denom & ~values))
                row[key + "_prevalence"] = (
                    float(pos / (pos + neg)) if min(pos, neg) >= MIN_CELL else None
                )
            tiles.append(row)
    write(
        out / "shared_geometry_aggregate.json",
        {
            "analysis_id": "shared_patch_k5_large_tile_umap_geometry",
            "reference_fit_reproduced": True,
            "embedding": {
                "algorithm": "UMAP",
                "seed": 42,
                "n_neighbors": 30,
                "min_dist": 0.3,
                "bins_per_axis": 12,
                "coordinates_serialized": False,
            },
            "tiles": tiles,
            "tile_semantics": "Dominant profile/resistance/region only when the displayed category has >=30 conditions; HIV/smear fractions require >=30 in both categories. Bins with fewer than30 are omitted.",
            "profile_order_risk_ranks": TABLE_RANKS,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dua-acknowledge", action="store_true", required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Reviewer source root containing src/ctb, scripts and manuscript/aggregate_results",
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "source_data",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "reference", "validation", "transport"],
        default="all",
    )
    args = parser.parse_args()
    os.umask(0o077)
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    output_names = {
        "reference": ["shared_context_atlas.json", "shared_geometry_aggregate.json"],
        "validation": ["shared_oof_clinical_validation.json"],
        "transport": ["shared_internal_transport.json"],
    }
    requested = list(output_names) if args.stage == "all" else [args.stage]
    if any((args.output_dir / n).exists() for s in requested for n in output_names[s]):
        raise FileExistsError(
            "Requested aggregate outputs already exist; use a fresh output directory"
        )
    for filename, digest in HASHES.items():
        assert (
            hashlib.sha256((args.raw_dir / filename).read_bytes()).hexdigest() == digest
        ), "Governed input hash mismatch"
    cfg = LockedValidationConfig()
    cases = clean_patient_cases(load_patient_cases(args.raw_dir))
    regimens = clean_regimens(load_regimens(args.raw_dir))
    bio = load_biochemistry(args.raw_dir)
    data = build_schema6_raw_validation_data(cases, regimens, bio, cfg)
    if args.stage in ["all", "reference"]:
        reference(cases, data, cfg, args.output_dir)
    if args.stage in ["all", "validation"]:
        validation(data, cfg, args.output_dir)
    if args.stage in ["all", "transport"]:
        transport(cases, data, cfg, args.output_dir)


if __name__ == "__main__":
    try:
        if "--help" in sys.argv or "-h" in sys.argv:
            main()
            raise SystemExit(0)
        with (
            open(os.devnull, "w") as sink,
            contextlib.redirect_stdout(sink),
            contextlib.redirect_stderr(sink),
        ):
            main()
        print(json.dumps({"status": "complete", "private_values_printed": False}))
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "private_values_printed": False,
                }
            )
        )
        raise SystemExit(1) from None
