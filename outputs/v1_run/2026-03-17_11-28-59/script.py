#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone runner for a prompts JSON file.

Input JSON format:
{
  "bio_1129": "The 1st token has \"iki\" as its lemma ...",
  "bio_1201": "The 1st token has \"gerçi\" as its lemma ..."
}

What it does:
- reads one prompts JSON file
- sends each prompt string directly to OpenAI Responses API
- uses model gpt-4o
- saves outputs keyed by the same sent_id
- resumes from prior runs if interrupted

Example usage:
python3 run_gpt4o_prompts.py \
  --prompts-file prompts_v1.json \
  --api-key YOUR_OPENAI_API_KEY \
  --output-dir outputs \
  --experiment-name turkish_v1

Resume:
python3 run_gpt4o_prompts.py \
  --run-dir outputs/turkish_v1/2026-03-17_07-10-00 \
  --api-key YOUR_OPENAI_API_KEY

Dry run:
python3 run_gpt4o_prompts.py \
  --prompts-file prompts_v1.json \
  --dry-run \
  --preview-only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


API_URL = "https://api.openai.com/v1/responses"
MODEL_NAME = "gpt-4o"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="Path to JSON file mapping sent_id -> prompt string",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Existing run directory to resume",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key. If omitted, OPENAI_API_KEY environment variable is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base output directory for new runs",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="prompt_run",
        help="Folder name for this experiment",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N prompts",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to wait between successful requests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="How many times to retry a failed request",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional temperature to send to the API",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Optional system-style instructions to send with every prompt",
    )
    parser.add_argument(
        "--note",
        type=str,
        default=None,
        help="Optional note stored in metadata",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the prompts file and write run metadata, but do not call the API",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="With --dry-run, print the first prompt and exit",
    )

    return parser.parse_args()


def get_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("run_gpt4o_prompts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path, default: Any) -> Any:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_prompts(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Prompts file must be a JSON object mapping sent_id -> prompt string.")

    cleaned: Dict[str, str] = {}
    for sent_id, prompt in data.items():
        if not isinstance(sent_id, str):
            raise ValueError(f"Invalid key {sent_id!r}: prompt IDs must be strings.")
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt for {sent_id!r} must be a string.")
        prompt = prompt.strip()
        if not prompt:
            raise ValueError(f"Prompt for {sent_id!r} is empty.")
        cleaned[sent_id] = prompt

    return cleaned

def wrap_prompt(test_prompt: str) -> str:
    return f"""Your task is to reconstruct a Turkish sentence using its morphosyntactic features in Universal Dependencies style annotations. First, I will provide the morphosyntactic features of an example sentence along with its corresponding reconstruction. Then, you will be given the morphosyntactic features of another sentence, and you will reconstruct it yourself. Your answer should be as follows:
{{"original_form": "<SENTENCE>"}}


Below are the morphosyntactic features of the sample Turkish sentence in Universal Dependencies style, consisting of the following {{6}} lemmas. The morphosyntactic features are provided for each token.

1st token's lemma is "insan", its part of speech is noun, its case is nominative, its number is plural number, its person is third person, and it depends on the 4th token with the relation of nominal subject.
2nd token's lemma is "yine", its part of speech is adverb, and it depends on the 4th token with the relation of adverbial modifier.
3rd token's lemma is "o", its part of speech is pronoun, its case is accusative, its number is plural number, its person is third person, its pronominal type is personal, and it depends on the 4th token with the relation of direct object.
4th token's lemma is "tercih", its part of speech is noun, its case is nominative, its number is singular number, its person is third person, and it's the root token.
5th token's lemma is "et", its part of speech is verb, its aspect is imperfect aspect, its number is singular number, its person is third person, it is positive, its tense is present tense, and it depends on the 4th token with the relation of light verb construction.
6th token's lemma is ".", its part of speech is punctuation, and it depends on the 4th token with the relation of punctuation.

Your output for the reconstruction of this morphosyntactic annotation should be as follows:
{"original_form": "İnsanlar yine onları tercih ediyor."}


Below are the morphosyntactic features of another Turkish sentence in Universal Dependencies style, consisting of the following {len(test_prompt.splitlines())} lemmas. The morphosyntactic features are provided for each token. Now, you will analyze it and provide the reconstruction of the sentence. Please include all the tokens in your answer in the correct order.

{test_prompt}

Now give me the reconstruction of this morphosyntactic annotation in the desired format."""

def build_run_dir(base_output_dir: Path, experiment_name: str) -> Path:
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return base_output_dir / experiment_name / now

def extract_json_only(text: str) -> str:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return text

def extract_output_text(response_json: Dict[str, Any]) -> str:
    """
    Try the simplest documented field first, then fall back to walking output items.
    """
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    collected = []
    for item in response_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text", "")
                if text:
                    collected.append(text)

    joined = "\n".join(part.strip() for part in collected if part and part.strip()).strip()
    return joined


def call_openai_responses_api(
    api_key: str,
    prompt: str,
    timeout: int,
    max_retries: int,
    instructions: str | None = None,
    temperature: float | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,
        "input": prompt,
    }

    if instructions is not None:
        payload["instructions"] = instructions

    if temperature is not None:
        payload["temperature"] = temperature

    data = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)

        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            last_error = RuntimeError(
                f"HTTPError {e.code}: {body[:2000]}"
            )

            # Retry on rate limit and server errors
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                backoff = min(2 ** (attempt - 1), 30)
                time.sleep(backoff)
                continue
            raise last_error

        except urllib.error.URLError as e:
            last_error = RuntimeError(f"URLError: {e}")
            if attempt < max_retries:
                backoff = min(2 ** (attempt - 1), 30)
                time.sleep(backoff)
                continue
            raise last_error

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                backoff = min(2 ** (attempt - 1), 30)
                time.sleep(backoff)
                continue
            raise

    raise RuntimeError(f"API call failed after {max_retries} retries: {last_error}")


def main() -> None:
    args = get_args()

    # Resume mode
    if args.run_dir is not None:
        run_dir = args.run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        md_path = run_dir / "md.json"
        if not md_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {md_path}")

        md = read_json(md_path, None)
        if md is None:
            raise ValueError(f"Could not read metadata from {md_path}")

        prompts_file = Path(md["prompts_file"])
        prompts = load_prompts(prompts_file)
        experiment_name = md.get("experiment_name", "prompt_run")

    # New run mode
    else:
        if args.prompts_file is None:
            raise ValueError("For a new run, you must provide --prompts-file.")

        prompts_file = args.prompts_file
        prompts = load_prompts(prompts_file)
        experiment_name = args.experiment_name
        run_dir = build_run_dir(args.output_dir, experiment_name)
        ensure_dir(run_dir)

        md = {
            "created_at": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "experiment_name": experiment_name,
            "prompts_file": str(prompts_file.resolve()),
            "run_dir": str(run_dir.resolve()),
            "model": MODEL_NAME,
            "api_url": API_URL,
            "prompt_count": len(prompts),
            "sleep": args.sleep,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
            "temperature": args.temperature,
            "instructions": args.instructions,
            "note": args.note,
        }
        write_json(run_dir / "md.json", md)

        # Save a copy of this script for reproducibility
        script_copy = run_dir / "script.py"
        script_copy.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    if args.limit is not None:
        prompts = dict(list(prompts.items())[:args.limit])

    ensure_dir(run_dir)
    logger = get_logger(run_dir / "run.log")

    prompts_used_path = run_dir / "prompts_used.json"
    outputs_path = run_dir / "outputs.json"
    errors_path = run_dir / "errors.json"

    write_json(prompts_used_path, prompts)

    existing_outputs = read_json(outputs_path, {})
    existing_errors = read_json(errors_path, {})

    if not isinstance(existing_outputs, dict):
        existing_outputs = {}
    if not isinstance(existing_errors, dict):
        existing_errors = {}

    done_ids = set(existing_outputs.keys())
    remaining = {sid: prompt for sid, prompt in prompts.items() if sid not in done_ids}

    print(f"Experiment: {experiment_name}")
    print(f"Run dir: {run_dir}")
    print(f"Total prompts: {len(prompts)}")
    print(f"Already completed: {len(done_ids)}")
    print(f"Remaining: {len(remaining)}")

    logger.info("Run started")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Total prompts: {len(prompts)}")
    logger.info(f"Already completed: {len(done_ids)}")
    logger.info(f"Remaining: {len(remaining)}")

    if args.dry_run:
        print("Dry run complete. No API calls made.")
        if args.preview_only and prompts:
            first_id = next(iter(prompts))
            print("\n" + "=" * 80)
            print(f"FIRST PROMPT ID: {first_id}")
            print("=" * 80)
            print(prompts[first_id])
            print("=" * 80)
        return

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found. Use --api-key or set OPENAI_API_KEY.")

    processed_now = 0

    for idx, (sent_id, prompt) in enumerate(remaining.items(), start=1):
        print(f"Processing {idx}/{len(remaining)}: {sent_id}")
        logger.info(f"Processing {idx}/{len(remaining)}: {sent_id}")

        try:
            response_json = call_openai_responses_api(
                api_key=api_key,
                prompt=wrap_prompt(prompt),
                timeout=args.timeout,
                max_retries=args.max_retries,
                instructions=args.instructions,
                temperature=args.temperature,
            )

            output_text = extract_output_text(response_json)
            output_text = extract_json_only(output_text)
            if not output_text:
                raise RuntimeError(f"No text output returned for {sent_id}.")

            existing_outputs[sent_id] = output_text
            write_json(outputs_path, existing_outputs)

            # Optional raw response archive per item
            raw_dir = run_dir / "raw_responses"
            ensure_dir(raw_dir)
            write_json(raw_dir / f"{sent_id}.json", response_json)

            processed_now += 1
            logger.info(f"Success: {sent_id}")

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception as e:
            existing_errors[sent_id] = str(e)
            write_json(errors_path, existing_errors)
            logger.exception(f"Failed: {sent_id}")
            print(f"Failed: {sent_id} -> {e}")

    print(f"Newly processed: {processed_now}")
    print(f"Total saved outputs: {len(existing_outputs)}")

    missing = [sid for sid in prompts if sid not in existing_outputs]
    if missing:
        print(f"Still missing: {len(missing)}")
    else:
        print("All prompts are done.")

    logger.info(f"Newly processed: {processed_now}")
    logger.info(f"Total saved outputs: {len(existing_outputs)}")
    logger.info("Run finished")


if __name__ == "__main__":
    main()