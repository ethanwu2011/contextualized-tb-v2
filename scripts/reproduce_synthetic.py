"""Reproduce the public PATCH pipeline using synthetic data only.

The real TB Portals export is never read by this runner. It forces ``CTB_RAW_DIR`` to the dedicated
synthetic directory, builds the endpoint-explicit matrices, and fits one small smoke-test model.
All resulting values are demonstrations of execution and have no biological interpretation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from ctb.config import PROJECT_ROOT


def _run(script: str, env: dict[str, str], *args: str) -> None:
    """Run a pipeline stage from the repository root."""
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / script), *args]
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def main() -> int:
    """Generate synthetic inputs and run the endpoint-explicit smoke path."""
    parser = argparse.ArgumentParser(description="Run PATCH on synthetic data only.")
    parser.add_argument("--n", type=int, default=300, help="number of synthetic conditions")
    parser.add_argument("--seed", type=int, default=20230815)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if not 50 <= args.n <= 5000:
        raise SystemExit("--n must be between 50 and 5000 for the public smoke run")
    if args.epochs < 1:
        raise SystemExit("--epochs must be positive")

    base_env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    base_env["PYTHONPATH"] = src_path + os.pathsep + base_env.get("PYTHONPATH", "")

    # Keep all generated files under the configured data root. This preserves the normal clone
    # defaults while allowing a reviewer or CI job to point the complete smoke run at a scratch
    # directory with CTB_DATA_DIR. Results and logs follow that root unless explicitly overridden.
    data_dir = os.environ.get("CTB_DATA_DIR", str(PROJECT_ROOT / "data"))
    data_root = os.path.abspath(os.path.expanduser(data_dir))
    synthetic_dir = os.path.join(data_root, "raw", "synthetic")
    base_env["CTB_DATA_DIR"] = data_root
    base_env.setdefault("CTB_RESULTS_DIR", os.path.join(os.path.dirname(data_root), "results"))
    base_env.setdefault("CTB_LOG_DIR", os.path.join(os.path.dirname(data_root), "logs"))
    os.makedirs(synthetic_dir, exist_ok=True)

    # The generator compares its destination with CTB_RAW_DIR to prevent accidental overwrites.
    # Generate first with that variable unset, then point the ETL stages at the synthetic directory.
    generate_env = base_env.copy()
    generate_env.pop("CTB_RAW_DIR", None)

    _run(
        "00b_make_synthetic_data.py",
        generate_env,
        "--n",
        str(args.n),
        "--seed",
        str(args.seed),
        "--out",
        synthetic_dir,
    )
    env = base_env.copy()
    env["CTB_RAW_DIR"] = synthetic_dir
    _run("01_build_dataset.py", env)
    _run("05c_build_unfavorable_matrices.py", env)
    _run("00d_public_smoke_model.py", env, "--epochs", str(args.epochs))
    print("Synthetic PATCH smoke run completed. All outputs are synthetic and non-biological.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
