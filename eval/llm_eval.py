import json
import re
import time
import argparse
import logging
import requests
import os
from pathlib import Path
from typing import Optional, Any

# --- Constants ---

API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_RETRIES = 5
BACKOFF_FACTOR_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 60

USER_PROMPT_TEMPLATE = """
Question: {question}
Ground Truth Answers: {gold_answer}
Model Response: {model_response}

Evaluation Instructions:
You are an AI assistant tasked with evaluating the correctness of model responses question, and ground truth answer. Your judgment should follow these principles:
1. Consider the question, and ground truth answer holistically before evaluating the model's response.
2. Your decision should be strictly **Yes or No**, based on whether the model's response is factually accurate and aligns with the ground truth answer.
3. If the model response is a more specific form which includes the ground truth answer, it is correct.
4. If the model response includes all key information but adds minor details, it is correct as long as the extra details are factually correct.
5. If the model response contradicts, modifies, or omits critical parts of the answer, it is incorrect.
6. For numerical values, ensure correctness even when presented in different units.
7. For names, check for first and last name correctness. If the middle name is extra but correct, consider it correct.
8. For yes/no questions, the response must exactly match "Yes" or "No" to be correct.
Evaluate whether the Model Response is correct based Question, and Ground Truth Answer. Follow the predefined judgment rules and provide a clear Yes/No answer along with a justification.

Output Format:
<reason>Detailed reasoning following the evaluation principles.</reason>
<judge>Yes/No</judge>
"""


def parse_judge_response(text: str) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    judge_match = re.search(r"<judge>([\s\S]*?)</judge>", text, re.IGNORECASE)
    reason_match = re.search(r"<reason>([\s\S]*?)</reason>", text, re.IGNORECASE)
    judge_score = judge_match.group(1).strip() if judge_match else None
    judge_reason = reason_match.group(1).strip() if reason_match else None
    if not judge_score:
        logging.warning(f"Could not parse <judge> tag from response: {text}")
    return judge_score, judge_reason


def call_llm_judge(session: requests.Session, user_prompt: str, model_name: str = MODEL_NAME) -> Optional[str]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(
                API_URL,
                headers=headers,
                data=json.dumps(payload),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
            text_part = result.get("choices", [{}])[0].get("message", {}).get("content")
            if text_part:
                return text_part
            logging.warning(f"API response missing expected text data: {result}")
            return None
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 429 or response.status_code >= 500:
                delay = BACKOFF_FACTOR_SECONDS * (2 ** attempt)
                logging.warning(f"API Error: {http_err}. Retrying in {delay}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
            else:
                logging.error(f"Non-retriable HTTP error: {http_err} - Response: {response.text}")
                return None
        except requests.exceptions.RequestException as req_err:
            delay = BACKOFF_FACTOR_SECONDS * (2 ** attempt)
            logging.warning(f"Request error: {req_err}. Retrying in {delay}s... (Attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay)
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return None
    logging.error(f"Failed to call API after {MAX_RETRIES} retries.")
    return None


def extract_last_reasoning_record(total_pred: str) -> str:
    """Extract the last reasoning record from Trajectory.all_reasoning() output.

    Format produced by Trajectory.all_reasoning():
        "Reasoning Record #1:\\n{reasoning}\\n\\nReasoning Record #2:\\n{reasoning}..."
    """
    if not total_pred:
        return ""
    # Split on "Reasoning Record #N:" optionally followed by newline
    records = re.split(r"Reasoning Record #\d+:\n?", total_pred)
    for record in reversed(records):
        stripped = record.strip()
        if stripped:
            return stripped
    return ""


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_data(input_file: str, output_file: str, eval_record: bool = False) -> None:
    logging.info(f"Starting processing for '{input_file}'...")

    try:
        rows = load_jsonl(input_file)
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at '{input_file}'")
        return
    except Exception as e:
        logging.error(f"Error reading input file: {e}")
        return

    logging.info(f"Found {len(rows)} rows to process.")

    # Resume: load already-processed indices from output file
    processed_rows: list[dict[str, Any]] = []
    processed_indices: set[int] = set()
    if Path(output_file).exists():
        try:
            existing = load_jsonl(output_file)
            processed_rows = existing
            processed_indices = {int(r["original_index"]) for r in existing if "original_index" in r}
            logging.info(f"Resuming from {len(processed_rows)} processed rows found in '{output_file}'.")
        except Exception as e:
            logging.warning(f"Could not load existing output file '{output_file}'. Starting from scratch. Error: {e}")

    with requests.Session() as session:
        for i, row in enumerate(rows):
            if i in processed_indices:
                continue

            question = row.get("question", "")
            gold_answer = row.get("gold_answer", row.get("answer", ""))

            if eval_record:
                model_response = extract_last_reasoning_record(row.get("total_pred", ""))
            else:
                model_response = row.get("prediction", row.get("answer", ""))

            user_prompt = USER_PROMPT_TEMPLATE.format(
                question=question,
                gold_answer=gold_answer,
                model_response=model_response,
            )

            logging.info(f"Processing row {i + 1}/{len(rows)}...")
            raw_response = call_llm_judge(session, user_prompt, model_name=MODEL_NAME)
            print(raw_response)

            judge_score, judge_reason = parse_judge_response(raw_response)

            new_row = row.copy()
            new_row["judge_score"] = judge_score
            new_row["judge_reason"] = judge_reason
            new_row["original_index"] = i
            processed_rows.append(new_row)

            if i % 10 == 0:
                save_jsonl(output_file, processed_rows)

    save_jsonl(output_file, processed_rows)
    logging.info(f"Finished processing. Total processed rows: {len(processed_rows)}.")


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------


def calculate_score(output_path: str) -> None:
    logging.info(f"Calculating score for '{output_path}'...")
    try:
        rows = load_jsonl(output_path)
    except FileNotFoundError:
        logging.error(f"Error: Processed file not found at '{output_path}'")
        return
    except Exception as e:
        logging.error(f"Error reading processed file: {e}")
        return

    scores = [str(r.get("judge_score") or "").strip().lower() for r in rows]
    yes_count = scores.count("yes")
    total_count = sum(1 for s in scores if s)

    if total_count == 0:
        logging.warning("No judge scores were found to calculate an accuracy.")
        return

    accuracy = (yes_count / total_count) * 100
    logging.info(f"Final Accuracy: {accuracy:.2f}% ({yes_count}/{total_count} correct)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not API_KEY:
        logging.error("OPENAI_API_KEY environment variable is not set!")
        return

    parser = argparse.ArgumentParser(description="Evaluate a PMSR JSONL output using an LLM as a judge.")
    parser.add_argument("--jsonl", required=True, help="Path to the input JSONL file produced by eval/main.py.")
    parser.add_argument("--eval-record", action="store_true",
                        help="Evaluate the last reasoning record instead of the final prediction.")
    args = parser.parse_args()

    output_base = str(Path(args.jsonl).with_suffix(""))
    if args.eval_record:
        output_path = f"{output_base}_llm_eval_record.jsonl"
    else:
        output_path = f"{output_base}_llm_eval.jsonl"

    process_data(args.jsonl, output_path, args.eval_record)
    calculate_score(output_path)


if __name__ == "__main__":
    main()
