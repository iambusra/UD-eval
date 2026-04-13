# UD-Eval: LLM Output Evaluation Framework

Evaluate Large Language Model (LLM) performance on reconstructing Turkish sentences from Universal Dependencies (UD) annotations.

## Overview

This repository contains tools to assess how well LLMs can reconstruct surface-level Turkish sentences when given UD-style morphosyntactic annotations. The LLM is prompted with detailed linguistic descriptions extracted from CoNLL-U formatted data and must generate the original sentence text. The generated outputs are then compared against the gold standard using multiple similarity metrics.

## Project Workflow

1. **Generate Prompts** (`prompt_generator.py`): Convert CoNLL-U linguistic annotations into natural language prompts describing each token's morphosyntactic properties.

2. **Run LLM Inference** (`run_llm_experiment.py`): Send the generated prompts to OpenAI's GPT-4o API and collect LLM responses.

3. **Compare Outputs** (`compare_llm_outputs.py`): Evaluate LLM outputs by comparing generated surface forms against the gold standard using character-level and lemma-level similarity metrics.

## Repository Structure

```
├── prompt_generator.py              # Converts CoNLL-U to natural language prompts
├── run_llm_experiment.py            # Runs LLM inference on prompts
├── compare_llm_outputs.py           # Compares LLM outputs with gold standard
│
├── sentences_errors.conllu          # CoNLL-U data with error annotations
├── sentences_fixed.conllu           # CoNLL-U data with corrected annotations
│
├── sentences_errors_prompts.json    # Generated prompts from error annotations
├── sentences_fixed_prompts.json     # Generated prompts from fixed annotations
│
├── errors_outputs.json              # LLM outputs for error test set
├── fixed_outputs.json               # LLM outputs for fixed test set
│
├── outputs/                         # Experiment run directories
│   └── v1_run/
│       ├── 2026-03-17_11-28-59/    # Run with timestamp
│       ├── 2026-03-17_11-29-52/    # Run with raw_responses/
│       └── 2026-03-17_11-32-13/    # Run with raw_responses/
│
├── comparison_results/              # Evaluation results
│   ├── detailed_comparison.csv      # Per-sentence evaluation metrics
│   └── summary.json                 # Aggregate statistics
│
└── additional experiments/           # Supplementary experimental data
```

## Data Format

### CoNLL-U Files
Input files are in [CoNLL-U format](https://universaldependencies.org/format.html):
- Each token is annotated with lemma, POS tag, morphological features, and dependency relations
- Turkish language-specific morphological features (case, aspect, voice, evidentiality, etc.)
- Sentence-level metadata includes `sent_id` and `text` (the gold standard surface form)

Example:
```
# sent_id = bio_1129
# text = İkinci kapıdan da çıkardı.
1	İkinci	iki	NUM	ANum	NumType=Ord	2	nummod	_	_
2	kapıdan	kapı	NOUN	_	Case=Abl|Number=Sing|Person=3	4	obl	_	_
...
```

### Prompt JSON Files
Structure: `{ "sent_id": "natural language description of annotation" }`

Prompts convert annotations into human-readable descriptions:
```json
{
  "bio_1129": "The 1st token has \"iki\" as its lemma and is an ordinal numeral...",
  "bio_1201": "The 1st token has \"gerçi\" as its lemma and is an adverb..."
}
```

### LLM Output JSON Files
Structure: `{ "sent_id": "{\"original_form\": \"surface text\"}" }`

LLM responses contain the reconstructed surface form:
```json
{
  "bio_1129": "{\"original_form\": \"İkinci kapıdan da çıkardı.\"}",
  "bio_1201": "{\"original_form\": \"Gerçi şimdilik beni kullandıkları yok ama...\"}"
}
```

## Usage

### Step 1: Generate Prompts

```bash
python3 prompt_generator.py \
  --conllu sentences_fixed.conllu \
  --output-file sentences_fixed_prompts.json
```

**Options:**
- `--conllu`: Path to CoNLL-U input file
- `--output-file`: Path to save generated prompts JSON

### Step 2: Run LLM Experiments

```bash
python3 run_llm_experiment.py \
  --prompts-file sentences_fixed_prompts.json \
  --api-key YOUR_OPENAI_API_KEY \
  --output-dir outputs \
  --experiment-name turkish_v1
```

**Options:**
- `--prompts-file`: Path to prompts JSON file
- `--api-key`: OpenAI API key
- `--output-dir`: Directory to save experiment outputs
- `--experiment-name`: Name for this experiment run
- `--dry-run`: Test without calling the API
- `--run-dir`: Resume from a previous run (provide the run directory path)

Outputs are saved under `output-dir/experiment-name/<timestamp>/` with:
- `outputs.json`: LLM responses
- `prompts_used.json`: Actual prompts sent to the API
- `md.json`: Metadata (timestamps, counts, etc.)
- `raw_responses/`: Individual JSON files per sentence ID

### Step 3: Compare Outputs

```bash
python3 compare_llm_outputs.py \
  --conllu sentences_fixed.conllu \
  --fixed-json fixed_outputs.json \
  --error-json errors_outputs.json \
  --output-dir comparison_results
```

**Options:**
- `--conllu`: CoNLL-U file with gold standard sentences
- `--fixed-json`: LLM outputs for fixed annotations
- `--error-json`: LLM outputs for error annotations
- `--output-dir`: Directory for results

Generates:
- `detailed_comparison.csv`: Per-sentence metrics (character similarity, lemma similarity, exact match)
- `summary.json`: Aggregate statistics

## Evaluation Metrics

The comparison script computes:

- **Character-level Similarity**: Normalized Levenshtein distance between surface forms
- **Lemma-level Similarity**: Similarity based on lemmatized forms (heuristic-based for Turkish)
- **Exact Match**: Binary indicator of perfect reconstruction
- **Unicode Normalization**: Handles Turkish-specific diacritics and composition forms
- **Quote & Punctuation Normalization**: Accounts for common text variation

Text preprocessing includes:
- Unicode NFC normalization
- Lowercasing for comparison
- Whitespace normalization
- Quote character standardization
- Apostrophe normalization

## Experiment Runs

Completed experiments are organized by timestamp in `outputs/v1_run/`:

Each run contains:
- `outputs.json`: Full LLM responses
- `prompts_used.json`: Actual system and user prompts sent to GPT-4o
- `md.json`: Run metadata (model, token counts, error logs)
- `raw_responses/`: Individual response files per sentence ID
- `errors.json` (optional): API or processing errors

## Requirements

- Python 3.7+
- OpenAI API key (for `run_llm_experiment.py`)
- Required packages: `requests` (API calls)

## Example Workflow

```bash
# 1. Generate prompts from annotations
python3 prompt_generator.py \
  --conllu sentences_fixed.conllu \
  --output-file prompts.json

# 2. Test with dry run
python3 run_llm_experiment.py \
  --prompts-file prompts.json \
  --api-key sk-... \
  --output-dir outputs \
  --experiment-name test_run \
  --dry-run

# 3. Run full experiment
python3 run_llm_experiment.py \
  --prompts-file prompts.json \
  --api-key sk-... \
  --output-dir outputs \
  --experiment-name test_run

# 4. Compare results
python3 compare_llm_outputs.py \
  --conllu sentences_fixed.conllu \
  --fixed-json outputs/test_run/2026-03-17_11-29-52/outputs.json \
  --error-json errors_outputs.json \
  --output-dir comparison_results
```

## Language Focus

- **Target Language**: Turkish
- **Annotation Standard**: Universal Dependencies
- **Task**: Surface form reconstruction from UD annotations
- **Test Sets**: Two variants (error-annotated and corrected annotations)

## License

See [LICENSE](LICENSE) file.

## Citation

If you use this framework in your research, please cite the project appropriately.
