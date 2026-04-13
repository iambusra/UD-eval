#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare_llm_outputs.py

What this script does
---------------------
1. Reads a CoNLL-U file and builds a dictionary:
       sent_id -> gold sentence text

2. Reads two LLM output JSON files:
       - one for errorful annotations
       - one for corrected/fixed annotations

   Expected format of each JSON file:
   {
       "bio_1129": "{\"original_form\": \"İki kapıdan da çıkmış.\"}",
       "bio_1201": "{\"original_form\": \"...\"}"
   }

   It also supports values that are already parsed dicts:
   {
       "bio_1129": {"original_form": "İki kapıdan da çıkmış."}
   }

3. Normalizes text before comparison:
       - Unicode normalization
       - lowercasing
       - quote normalization
       - apostrophe normalization
       - punctuation stripping
       - whitespace cleanup

4. Computes:
       - character-level similarity
       - lemma-level similarity

5. Writes:
       - per-sentence detailed CSV
       - summary JSON

Notes
-----
- Lemma-level similarity here is approximation-based and standalone.
- Since you asked for a script that stands on its own, this version does NOT
  depend on external Turkish NLP packages.
- Lemma normalization is heuristic:
    * lowercase
    * strip punctuation
    * remove common clitic/apostrophe effects
    * optionally remove a few very common Turkish suffix patterns
- If later you want true morphological lemmatization, I can give you a second
  version using a Turkish NLP library. But this one will run as-is.

Example usage
-------------
python3 compare_llm_outputs.py \
  --conllu sentences_fixed.conllu \
  --fixed-json fixed_outputs.json \
  --error-json errors_outputs.json \
  --output-dir comparison_results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Any


PUNCT_REGEX = re.compile(r"[^\w\s]", flags=re.UNICODE)
MULTISPACE_REGEX = re.compile(r"\s+")
TOKEN_REGEX = re.compile(r"\b\w+\b", flags=re.UNICODE)

# Very light heuristic suffix stripping for Turkish-like surface matching.
# Ordered roughly from longer/more specific to shorter/more general.
COMMON_SUFFIXES = [
    "larımızdan", "lerimizden", "larımızda", "lerimizde",
    "larımızı", "lerimizi", "larımız", "lerimiz",
    "larının", "lerinin", "larına", "lerine",
    "larını", "lerini", "lardan", "lerden",
    "larda", "lerde", "ların", "lerin",
    "lara", "lere", "ları", "leri",
    "ımız", "imiz", "umuz", "ümüz",
    "ınız", "iniz", "unuz", "ünüz",
    "mız", "miz", "muz", "müz",
    "nız", "niz", "nuz", "nüz",
    "dan", "den", "tan", "ten",
    "dır", "dir", "dur", "dür",
    "tır", "tir", "tur", "tür",
    "mış", "miş", "muş", "müş",
    "yor", "acak", "ecek",
    "ınca", "ince", "unca", "ünce",
    "sın", "sin", "sun", "sün",
    "sınız", "siniz", "sunuz", "sünüz",
    "lar", "ler",
    "ın", "in", "un", "ün",
    "ı", "i", "u", "ü",
    "a", "e",
    "m", "n", "k",
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--conllu", type=Path, required=True, help="Path to gold .conllu file")
    parser.add_argument("--fixed-json", type=Path, required=True, help="Path to fixed outputs JSON")
    parser.add_argument("--error-json", type=Path, required=True, help="Path to errorful outputs JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("comparison_results"))
    parser.add_argument(
        "--strip-suffixes",
        action="store_true",
        help="Apply heuristic Turkish suffix stripping for lemma-level comparison",
    )

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_quotes(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
        "`": "'",
        "´": "'",
        "…": "...",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text


def normalize_text(text: str, strip_punct: bool = True) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = normalize_quotes(text)
    text = text.lower().strip()

    # Normalize apostrophe spacing patterns a bit
    text = text.replace(" '", "'").replace("' ", "'")

    if strip_punct:
        text = PUNCT_REGEX.sub(" ", text)

    text = MULTISPACE_REGEX.sub(" ", text).strip()
    return text


def tokenize_normalized(text: str) -> List[str]:
    if not text:
        return []
    return TOKEN_REGEX.findall(text)


def heuristic_lemma(token: str, strip_suffixes: bool = False) -> str:
    tok = normalize_text(token, strip_punct=True)

    if not tok:
        return ""

    if not strip_suffixes:
        return tok

    # heuristic suffix stripping
    for suf in COMMON_SUFFIXES:
        if tok.endswith(suf) and len(tok) > len(suf) + 1:
            return tok[: -len(suf)]

    return tok


def lemma_sequence(text: str, strip_suffixes: bool = False) -> List[str]:
    tokens = tokenize_normalized(text)
    lemmas = [heuristic_lemma(tok, strip_suffixes=strip_suffixes) for tok in tokens]
    return [x for x in lemmas if x]


def char_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def sequence_similarity(seq1: List[str], seq2: List[str]) -> float:
    return SequenceMatcher(None, seq1, seq2).ratio()


def jaccard_similarity(seq1: List[str], seq2: List[str]) -> float:
    s1, s2 = set(seq1), set(seq2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def multiset_jaccard(seq1: List[str], seq2: List[str]) -> float:
    c1, c2 = Counter(seq1), Counter(seq2)
    all_keys = set(c1) | set(c2)
    if not all_keys:
        return 1.0
    intersection = sum(min(c1[k], c2[k]) for k in all_keys)
    union = sum(max(c1[k], c2[k]) for k in all_keys)
    return intersection / union if union else 1.0


def f1_overlap(seq1: List[str], seq2: List[str]) -> float:
    c1, c2 = Counter(seq1), Counter(seq2)
    overlap = sum(min(c1[k], c2[k]) for k in set(c1) | set(c2))
    pred_total = sum(c2.values())
    gold_total = sum(c1.values())

    if gold_total == 0 and pred_total == 0:
        return 1.0
    if gold_total == 0 or pred_total == 0:
        return 0.0

    precision = overlap / pred_total
    recall = overlap / gold_total

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def parse_conllu_sentences(conllu_path: Path) -> Dict[str, str]:
    """
    Build sent_id -> sentence text.

    Priority:
    1. # text = ...
    2. reconstruct from token FORM column
    """
    sentences: Dict[str, str] = {}

    with conllu_path.open("r", encoding="utf-8") as f:
        current_sent_id = None
        current_text = None
        token_forms: List[Tuple[int, str]] = []

        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                if current_sent_id is not None:
                    if current_text is None:
                        # reconstruct from token forms
                        ordered = [form for _, form in sorted(token_forms, key=lambda x: x[0])]
                        current_text = " ".join(ordered).strip()
                        current_text = fix_spacing_before_punct(current_text)
                    sentences[current_sent_id] = current_text

                current_sent_id = None
                current_text = None
                token_forms = []
                continue

            if line.startswith("#"):
                if line.startswith("# sent_id ="):
                    current_sent_id = line.split("=", 1)[1].strip()
                elif line.startswith("# text ="):
                    current_text = line.split("=", 1)[1].strip()
                continue

            parts = line.split("\t")
            if len(parts) != 10:
                continue

            tok_id = parts[0]
            form = parts[1]

            # skip multiword tokens and empty nodes
            if "-" in tok_id or "." in tok_id:
                continue

            try:
                int_id = int(tok_id)
            except ValueError:
                continue

            token_forms.append((int_id, form))

        # catch last sentence if file doesn't end with blank line
        if current_sent_id is not None:
            if current_text is None:
                ordered = [form for _, form in sorted(token_forms, key=lambda x: x[0])]
                current_text = " ".join(ordered).strip()
                current_text = fix_spacing_before_punct(current_text)
            sentences[current_sent_id] = current_text

    return sentences


def fix_spacing_before_punct(text: str) -> str:
    # remove space before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # fix parentheses spacing
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    # fix quotes spacing
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r'"\s+', '" ', text)

    # collapse whitespace
    text = MULTISPACE_REGEX.sub(" ", text).strip()

    return text


def parse_model_output_value(value: Any) -> str:
    """
    Supports:
    1. string containing JSON: "{\"original_form\": \"...\"}"
    2. dict with original_form
    3. raw string fallback
    """
    if isinstance(value, dict):
        if "original_form" in value:
            return str(value["original_form"]).strip()
        return json.dumps(value, ensure_ascii=False)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return ""

        # Try parse as JSON object string
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and "original_form" in parsed:
                return str(parsed["original_form"]).strip()
        except json.JSONDecodeError:
            pass

        return stripped

    return str(value).strip()


def load_outputs(path: Path) -> Dict[str, str]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object mapping sent_id -> output")

    result: Dict[str, str] = {}
    for sent_id, value in data.items():
        result[str(sent_id)] = parse_model_output_value(value)
    return result


def compare_one(
    gold_text: str,
    pred_text: str,
    strip_suffixes: bool = False,
) -> Dict[str, Any]:
    gold_norm = normalize_text(gold_text, strip_punct=True)
    pred_norm = normalize_text(pred_text, strip_punct=True)

    gold_lemmas = lemma_sequence(gold_text, strip_suffixes=strip_suffixes)
    pred_lemmas = lemma_sequence(pred_text, strip_suffixes=strip_suffixes)

    return {
        "gold_raw": gold_text,
        "pred_raw": pred_text,
        "gold_norm": gold_norm,
        "pred_norm": pred_norm,
        "char_similarity": char_similarity(gold_norm, pred_norm),
        "lemma_seq_similarity": sequence_similarity(gold_lemmas, pred_lemmas),
        "lemma_jaccard": jaccard_similarity(gold_lemmas, pred_lemmas),
        "lemma_multiset_jaccard": multiset_jaccard(gold_lemmas, pred_lemmas),
        "lemma_f1": f1_overlap(gold_lemmas, pred_lemmas),
        "gold_lemma_count": len(gold_lemmas),
        "pred_lemma_count": len(pred_lemmas),
        "gold_lemmas": " ".join(gold_lemmas),
        "pred_lemmas": " ".join(pred_lemmas),
    }


def build_rows(
    gold_sentences: Dict[str, str],
    fixed_outputs: Dict[str, str],
    error_outputs: Dict[str, str],
    strip_suffixes: bool = False,
) -> List[Dict[str, Any]]:
    sent_ids = sorted(set(gold_sentences) | set(fixed_outputs) | set(error_outputs))
    rows: List[Dict[str, Any]] = []

    for sent_id in sent_ids:
        gold = gold_sentences.get(sent_id, "")
        fixed = fixed_outputs.get(sent_id, "")
        error = error_outputs.get(sent_id, "")

        fixed_cmp = compare_one(gold, fixed, strip_suffixes=strip_suffixes) if gold and fixed else None
        error_cmp = compare_one(gold, error, strip_suffixes=strip_suffixes) if gold and error else None

        row = {
            "sent_id": sent_id,
            "gold_text": gold,
            "fixed_output": fixed,
            "error_output": error,
            "in_gold": sent_id in gold_sentences,
            "in_fixed": sent_id in fixed_outputs,
            "in_error": sent_id in error_outputs,
        }

        if fixed_cmp:
            row.update({
                "fixed_char_similarity": fixed_cmp["char_similarity"],
                "fixed_lemma_seq_similarity": fixed_cmp["lemma_seq_similarity"],
                "fixed_lemma_jaccard": fixed_cmp["lemma_jaccard"],
                "fixed_lemma_multiset_jaccard": fixed_cmp["lemma_multiset_jaccard"],
                "fixed_lemma_f1": fixed_cmp["lemma_f1"],
                "fixed_gold_norm": fixed_cmp["gold_norm"],
                "fixed_pred_norm": fixed_cmp["pred_norm"],
                "fixed_gold_lemmas": fixed_cmp["gold_lemmas"],
                "fixed_pred_lemmas": fixed_cmp["pred_lemmas"],
            })
        else:
            row.update({
                "fixed_char_similarity": None,
                "fixed_lemma_seq_similarity": None,
                "fixed_lemma_jaccard": None,
                "fixed_lemma_multiset_jaccard": None,
                "fixed_lemma_f1": None,
                "fixed_gold_norm": "",
                "fixed_pred_norm": "",
                "fixed_gold_lemmas": "",
                "fixed_pred_lemmas": "",
            })

        if error_cmp:
            row.update({
                "error_char_similarity": error_cmp["char_similarity"],
                "error_lemma_seq_similarity": error_cmp["lemma_seq_similarity"],
                "error_lemma_jaccard": error_cmp["lemma_jaccard"],
                "error_lemma_multiset_jaccard": error_cmp["lemma_multiset_jaccard"],
                "error_lemma_f1": error_cmp["lemma_f1"],
                "error_gold_norm": error_cmp["gold_norm"],
                "error_pred_norm": error_cmp["pred_norm"],
                "error_gold_lemmas": error_cmp["gold_lemmas"],
                "error_pred_lemmas": error_cmp["pred_lemmas"],
            })
        else:
            row.update({
                "error_char_similarity": None,
                "error_lemma_seq_similarity": None,
                "error_lemma_jaccard": None,
                "error_lemma_multiset_jaccard": None,
                "error_lemma_f1": None,
                "error_gold_norm": "",
                "error_pred_norm": "",
                "error_gold_lemmas": "",
                "error_pred_lemmas": "",
            })

        # improvement columns
        if fixed_cmp and error_cmp:
            row["char_similarity_diff_fixed_minus_error"] = (
                row["fixed_char_similarity"] - row["error_char_similarity"]
            )
            row["lemma_seq_similarity_diff_fixed_minus_error"] = (
                row["fixed_lemma_seq_similarity"] - row["error_lemma_seq_similarity"]
            )
            row["lemma_f1_diff_fixed_minus_error"] = (
                row["fixed_lemma_f1"] - row["error_lemma_f1"]
            )
        else:
            row["char_similarity_diff_fixed_minus_error"] = None
            row["lemma_seq_similarity_diff_fixed_minus_error"] = None
            row["lemma_f1_diff_fixed_minus_error"] = None

        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    fixed_char = [r["fixed_char_similarity"] for r in rows if r["fixed_char_similarity"] is not None]
    error_char = [r["error_char_similarity"] for r in rows if r["error_char_similarity"] is not None]

    fixed_lemma_seq = [r["fixed_lemma_seq_similarity"] for r in rows if r["fixed_lemma_seq_similarity"] is not None]
    error_lemma_seq = [r["error_lemma_seq_similarity"] for r in rows if r["error_lemma_seq_similarity"] is not None]

    fixed_lemma_f1 = [r["fixed_lemma_f1"] for r in rows if r["fixed_lemma_f1"] is not None]
    error_lemma_f1 = [r["error_lemma_f1"] for r in rows if r["error_lemma_f1"] is not None]

    diffs_char = [
        r["char_similarity_diff_fixed_minus_error"]
        for r in rows
        if r["char_similarity_diff_fixed_minus_error"] is not None
    ]
    diffs_lemma_seq = [
        r["lemma_seq_similarity_diff_fixed_minus_error"]
        for r in rows
        if r["lemma_seq_similarity_diff_fixed_minus_error"] is not None
    ]
    diffs_lemma_f1 = [
        r["lemma_f1_diff_fixed_minus_error"]
        for r in rows
        if r["lemma_f1_diff_fixed_minus_error"] is not None
    ]

    summary = {
        "n_rows_total": len(rows),
        "n_rows_with_fixed_and_error_and_gold": len(diffs_char),

        "fixed_mean_char_similarity": safe_mean(fixed_char),
        "error_mean_char_similarity": safe_mean(error_char),

        "fixed_mean_lemma_seq_similarity": safe_mean(fixed_lemma_seq),
        "error_mean_lemma_seq_similarity": safe_mean(error_lemma_seq),

        "fixed_mean_lemma_f1": safe_mean(fixed_lemma_f1),
        "error_mean_lemma_f1": safe_mean(error_lemma_f1),

        "mean_char_similarity_diff_fixed_minus_error": safe_mean(diffs_char),
        "mean_lemma_seq_similarity_diff_fixed_minus_error": safe_mean(diffs_lemma_seq),
        "mean_lemma_f1_diff_fixed_minus_error": safe_mean(diffs_lemma_f1),

        "n_fixed_better_char": sum(1 for x in diffs_char if x > 0),
        "n_error_better_char": sum(1 for x in diffs_char if x < 0),
        "n_tie_char": sum(1 for x in diffs_char if x == 0),

        "n_fixed_better_lemma_seq": sum(1 for x in diffs_lemma_seq if x > 0),
        "n_error_better_lemma_seq": sum(1 for x in diffs_lemma_seq if x < 0),
        "n_tie_lemma_seq": sum(1 for x in diffs_lemma_seq if x == 0),

        "n_fixed_better_lemma_f1": sum(1 for x in diffs_lemma_f1 if x > 0),
        "n_error_better_lemma_f1": sum(1 for x in diffs_lemma_f1 if x < 0),
        "n_tie_lemma_f1": sum(1 for x in diffs_lemma_f1 if x == 0),
    }

    return summary


def main() -> None:
    args = get_args()
    ensure_dir(args.output_dir)

    gold_sentences = parse_conllu_sentences(args.conllu)
    fixed_outputs = load_outputs(args.fixed_json)
    error_outputs = load_outputs(args.error_json)

    rows = build_rows(
        gold_sentences=gold_sentences,
        fixed_outputs=fixed_outputs,
        error_outputs=error_outputs,
        strip_suffixes=args.strip_suffixes,
    )

    detailed_csv = args.output_dir / "detailed_comparison.csv"
    summary_json = args.output_dir / "summary.json"

    write_csv(rows, detailed_csv)
    summary = build_summary(rows)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Gold sentences loaded: {len(gold_sentences)}")
    print(f"Fixed outputs loaded: {len(fixed_outputs)}")
    print(f"Error outputs loaded: {len(error_outputs)}")
    print(f"Detailed CSV written to: {detailed_csv}")
    print(f"Summary JSON written to: {summary_json}")
    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()