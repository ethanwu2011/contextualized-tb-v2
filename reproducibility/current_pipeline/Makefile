.PHONY: all setup manuscript-smoke manuscript-dua aggregate-figures claim-audits-dua \
	brazil-audit-dua locked-validation-dca-dua public-file-manifest public-release-check phase1 \
	aggregate-checksums data build train figures test lint format type clean help
UV := uv run --frozen

help:
	@echo "Targets:"
	@echo "  setup             Create venv and sync the frozen dev environment"
	@echo "  all               Alias for verify (software, synthetic and aggregate checks)"
	@echo "  manuscript-smoke  Run tracked synthetic endpoint/S1-S2 workflow + provenance"
	@echo "  manuscript-dua    Full budgets; requires CTB_RAW_DIR + fresh CTB_DUA_OUTPUT_ROOT"
	@echo "  aggregate-figures Rebuild corrected figures from public aggregate JSON only"
	@echo "  claim-audits-dua  Rerun the corrected claim audit in a secure output directory"
	@echo "  brazil-audit-dua  Rebuild SINAN transport; also requires CTB_SINAN_SRC"
	@echo "  locked-validation-dca-dua  Rerun locked OOF validation and internal DCA"
	@echo "  public-file-manifest  Rebuild the staged-release manifest (needs the reviewer archive)"
	@echo "  aggregate-checksums  Rebuild hashes for reviewed aggregate JSON files"
	@echo "  public-release-check  Stage and scan the release (needs the reviewer archive)"
	@echo "  phase1            BLOCKED archived endpoint workflow (see remediation message)"
	@echo "  build             BLOCKED archived Phase 1 dataset writer"
	@echo "  train             BLOCKED analysis outside the release manifest"
	@echo "  figures           BLOCKED cached/legacy figure renderer"
	@echo "  test              Run pytest with coverage"
	@echo "  lint              ruff check + format --check"
	@echo "  type              mypy --strict on src"
	@echo "  clean             Remove generated outputs (keeps raw data)"

setup:
	uv venv --python 3.11
	uv sync --frozen --extra dev

all: verify

.PHONY: verify package-check
# In this repository folder, verify covers lint, types, unit tests and the synthetic workflow.
# package-check, aggregate-figures, public-file-manifest, public-release-check and the
# release-contract tests need the full reviewer archive (verify_package.py,
# PUBLIC_RELEASE_MANIFEST.json and the aggregate reports), which is not carried here. To check the
# public tree, run make public-release-check from the repository root.
verify: lint type test manuscript-smoke

package-check:
	$(UV) python verify_package.py

manuscript-smoke:
	$(UV) python scripts/run_manuscript.py --mode synthetic --reduced --output-root "$$(mktemp -d /tmp/patch-synthetic.XXXXXX)/run"

manuscript-dua:
	@test -n "$(CTB_DUA_OUTPUT_ROOT)" || \
		(echo "BLOCKED: set CTB_DUA_OUTPUT_ROOT to a fresh, secure output directory."; exit 2)
	$(UV) python scripts/run_manuscript.py --mode dua --output-root "$(CTB_DUA_OUTPUT_ROOT)"

aggregate-figures:
	$(UV) python verify_package.py --replay-aggregates

claim-audits-dua:
	@test -n "$(CTB_RAW_DIR)" || \
		(echo "BLOCKED: set CTB_RAW_DIR to the approved August 2023 TB Portals bundle."; exit 2)
	@test -n "$(CTB_DUA_OUTPUT_ROOT)" || \
		(echo "BLOCKED: set CTB_DUA_OUTPUT_ROOT to a fresh secure directory."; exit 2)
	$(UV) python scripts/prepare_audit_output_root.py \
		--output-root "$(CTB_DUA_OUTPUT_ROOT)" --project-root "$(CURDIR)" \
		--input-root "$(CTB_RAW_DIR)" --subdir aggregate --subdir simulation
	$(UV) python scripts/run_episode_endpoint_audit.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/episode_endpoint_audit.json"
	$(UV) python scripts/run_aggregate_sensitivities.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/sensitivity_and_support.json"
	$(UV) python scripts/run_architecture_composition_ablations.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/architecture_composition_ablations.json"
	$(UV) python scripts/run_shared_k_sensitivity.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/shared_k_sensitivity.json"
	$(UV) python scripts/run_shared_profile_full_pipeline_stability.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/shared_profile_full_pipeline_stability.json"
	$(UV) python scripts/run_shared_profile_reference_composition.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/shared_profile_reference_composition.json"
	$(UV) python scripts/run_internal_transport.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/internal_transport.json"
	$(UV) python scripts/run_expanded_simulation.py \
		--output-root "$(CTB_DUA_OUTPUT_ROOT)/simulation" \
		--aggregate-output \
		"$(CTB_DUA_OUTPUT_ROOT)/aggregate/expanded_simulation_benchmark.json"

brazil-audit-dua:
	@test -n "$(CTB_RAW_DIR)" || \
		(echo "BLOCKED: set CTB_RAW_DIR to the approved August 2023 TB Portals bundle."; exit 2)
	@test -n "$(CTB_SINAN_SRC)" || \
		(echo "BLOCKED: set CTB_SINAN_SRC to TUBEBR15.dbc through TUBEBR19.dbc."; exit 2)
	@test -n "$(CTB_DUA_OUTPUT_ROOT)" || \
		(echo "BLOCKED: set CTB_DUA_OUTPUT_ROOT to a fresh secure directory."; exit 2)
	$(UV) python scripts/prepare_audit_output_root.py \
		--output-root "$(CTB_DUA_OUTPUT_ROOT)" --project-root "$(CURDIR)" \
		--input-root "$(CTB_RAW_DIR)" --input-root "$(CTB_SINAN_SRC)" --subdir aggregate
	uv run --frozen --extra brazil python scripts/run_brazil_source_rebuild_audit.py \
		--dua-acknowledge --tb-raw-dir "$(CTB_RAW_DIR)" --sinan-src "$(CTB_SINAN_SRC)" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/brazil_source_rebuild_audit.json"

locked-validation-dca-dua:
	@test -n "$(CTB_RAW_DIR)" || \
		(echo "BLOCKED: set CTB_RAW_DIR to the approved August 2023 TB Portals bundle."; exit 2)
	@test -n "$(CTB_DUA_OUTPUT_ROOT)" || \
		(echo "BLOCKED: set CTB_DUA_OUTPUT_ROOT to a fresh secure aggregate directory."; exit 2)
	@test -n "$(CTB_PRIVATE_OUTPUT_ROOT)" || \
		(echo "BLOCKED: set CTB_PRIVATE_OUTPUT_ROOT to a separate fresh private directory."; exit 2)
	$(UV) python scripts/prepare_audit_output_root.py \
		--output-root "$(CTB_DUA_OUTPUT_ROOT)" --project-root "$(CURDIR)" \
		--input-root "$(CTB_RAW_DIR)" --input-root "$(CTB_PRIVATE_OUTPUT_ROOT)" \
		--subdir aggregate
	$(UV) python scripts/run_locked_residual_analysis.py --dua-acknowledge \
		--raw-dir "$(CTB_RAW_DIR)" --private-output-root "$(CTB_PRIVATE_OUTPUT_ROOT)" \
		--aggregate-output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/locked_nested_validation.json"
	$(UV) python scripts/run_internal_decision_curve.py --dua-acknowledge \
		--private-predictions \
		"$(CTB_PRIVATE_OUTPUT_ROOT)/dua_private/oof_predictions_DUA_PRIVATE.npz" \
		--output "$(CTB_DUA_OUTPUT_ROOT)/aggregate/internal_decision_curve.json"

public-file-manifest:
	$(UV) python verify_package.py --seal

aggregate-checksums:
	$(UV) python scripts/build_aggregate_checksums.py

public-release-check:
	@stage_parent="$$(mktemp -d /tmp/patch-public-release.XXXXXX)"; \
	bash scripts/sync_to_public.sh "$$stage_parent/release"; \
	bash "$$stage_parent/release/scripts/scan_public.sh" "$$stage_parent/release"
	$(UV) python verify_package.py --git-clone-check

data:
	$(UV) python scripts/00_download_data.py

build:
	@echo "BLOCKED: archived Phase 1 dataset/table outputs are outside the release manifest."
	@exit 2

train:
	@echo "BLOCKED: model training outside scripts/run_manuscript.py is not release-authorized."
	@exit 2

figures:
	@echo "BLOCKED: cached and legacy figures are not submission evidence."
	@exit 2

phase1:
	@echo "BLOCKED: archived treatment_intensification is unassessable in the audited schema."
	@echo "Do not fill missing labels. Use manuscript-smoke or adjudicate a valid endpoint first."
	@exit 2

ARCHIVE_ONLY_TESTS := \
	--deselect tests/test_manuscript_workflow.py::test_public_stager_is_fresh_target_only_and_scans_the_result \
	--deselect tests/test_release_contracts.py::test_public_manifest_names_corrected_endpoint_and_release_paths \
	--deselect tests/test_release_contracts.py::test_public_manifest_claims_match_checksummed_aggregate_reports

test:
	$(UV) pytest --cov=ctb --cov-report=term-missing -q $(ARCHIVE_ONLY_TESTS)

lint:
	$(UV) ruff check src tests scripts figures/release
	$(UV) ruff format --check src tests scripts figures/release

format:
	$(UV) ruff check --fix src tests scripts figures/release
	$(UV) ruff format src tests scripts figures/release

type:
	$(UV) mypy

clean:
	rm -rf results/figures/* results/metrics/* results/tables/* \
	       data/interim/* data/processed/* logs/*
