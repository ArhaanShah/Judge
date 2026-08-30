# Judge Swap Study - Agent Guidance

## Project Structure
- `src/` - Main source code (condition building, judging, analysis)
- `tests/` - Unit tests
- `configs/` - Configuration files (including legacy `configs/pilot.yaml`)
- `config/` - PRMBench-specific configurations (`pilot_prmbench_cohere.yaml`)
- `data/` - Source data and generated artifacts
- `results/` - Output from judge runs and analysis
- `artifacts/` - Frozen pilot manifests

## Key Commands
### Installation
```bash
python -m pip install -e ".[test]"  # Base + test dependencies
python -m pip install -e ".[test,pilot]"  # Plus Cohere/PRMBench deps
```

### Testing
```bash
python -m pytest  # Run all tests
```

### PRMBench/Cohere Pilot Workflow (from README)
1. Install dependencies: `python -m pip install -e ".[test,pilot]"`
2. Execute frozen stages in order:
   ```bash
   python -m src.data.prmbench_adapter --inspect
   python -m src.build_conditions --config config/pilot_prmbench_cohere.yaml --split smoke
   python -m src.validate_conditions --config config/pilot_prmbench_cohere.yaml --split smoke
   python -m src.run_judge --config config/pilot_prmbench_cohere.yaml --split smoke --provider cohere --confirm-trial-evaluation --available-call-budget 800
   
   python -m src.build_conditions --config config/pilot_prmbench_cohere.yaml --split main
   python -m src.validate_conditions --config config/pilot_prmbench_cohere.yaml --split main
   # Commit validated implementation and generated manifests, then:
   python -m src.freeze_pilot --config config/pilot_prmbench_cohere.yaml
   python -m src.run_judge --config config/pilot_prmbench_cohere.yaml --split main --provider cohere --resume --confirm-trial-evaluation --available-call-budget 800
   python -m src.analyze --config config/pilot_prmbench_cohere.yaml --split main
   ```

## Important Notes
- Live stages require `COHERE_API_KEY` present, explicit confirmation of trial/evaluation use, and ≥800 free calls
- Unit tests use mocked providers and make zero live requests
- Validation is a hard gate before judge calls (re-run immediately before)
- `run_judge` randomizes call order using saved seed, writes requests before sending, saves responses immediately, and resumes partially completed runs
- Choose new run ID for genuinely new runs
- Analysis writes to `results/` directory: summary CSV, Markdown tables, `go_kill.json`, and figures
- Raw logs remain under `results/raw/<run-id>/`

## Configuration Files
- `config/pilot_prmbench_cohere.yaml` - Main PRMBench/Cohere pilot config
- `configs/pilot.yaml` - Legacy generic fixture format config
- `config/prmbench_category_map.yaml` - PRMBench category mapping (update if needed)
- `config/prompts/pairwise_judge.txt` - Judge prompt template

## Data Locations
- Source items: `data/source/base_items.jsonl` (legacy) or `data/source/prmbench_rows.jsonl` (PRMBench)
- Generated conditions: `data/generated/conditions.jsonl` (legacy) or `data/processed/*_conditions.jsonl` (PRMBench)
- Runs: `data/runs/` (legacy) or `results/raw/` (PRMBench)
- Validation reports: `data/generated/validation_report.json` or `data/processed/*_validation_report.json`

## Python Modules
All functionality is accessible via `python -m src.<module>`:
- `src.data.prmbench_adapter` - PRMBench data inspection
- `src.build_conditions` - Build judge conditions
- `src.validate_conditions` - Validate conditions
- `src.run_judge` - Execute judge calls
- `src.freeze_pilot` - Freeze pilot configuration
- `src.analyze` - Analyze results
- `src.aggregate_pairs` - Aggregate pairs (legacy)