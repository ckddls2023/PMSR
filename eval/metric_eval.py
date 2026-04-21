"""Metrics for PMSR JSONL outputs.

Current runs store model output under ``trajectory``. Accuracy reads
``trajectory.final_answer`` and recall reads ``trajectory.all_knowledge``.
Legacy flat rows with ``prediction``, ``answer``, ``knowledge``, and
``total_pred`` remain readable for older outputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import string
import unicodedata
import numpy as np
from pathlib import Path
from typing import Any


bem_model = None
bem_tokenizer = None
bem_cls_id = None
bem_sep_id = None

_BEM_VOCAB_PATH = (
    "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt"
)
_BEM_MODEL_PATH = "https://tfhub.dev/google/answer_equivalence/bem/1"
_BEM_MAX_TOKENS = 512

_PUNCTUATION = string.punctuation + "‘’´`_"
_DIGIT_MAP = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "entailment": "yes",
    "true": "yes",
    "contradiction": "no",
    "false": "no",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PMSR JSONL predictions.")
    parser.add_argument("--jsonl", required=True, help="Path to PMSR prediction JSONL.")
    parser.add_argument(
        "--bem",
        action="store_true",
        help="Evaluate BEM accuracy instead of CEM accuracy.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first N predictions.",
    )
    parser.add_argument(
        "--iterative-recall",
        action="store_true",
        help="Print cumulative per-iteration recall and marginal gains.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_predictions(args: argparse.Namespace) -> list[dict[str, Any]]:
    predictions = load_jsonl(args.jsonl)
    print(f"Loaded {len(predictions)} predictions from JSONL: {args.jsonl}")
    return predictions


def preprocess_answer(answer: Any) -> str:
    text = str(answer).lower().replace("\n", " ").replace("\t", " ").strip()
    text = text.replace("<extra_id_0> ", "")
    text = "".join("" if char in _PUNCTUATION else char for char in text)
    text = re.sub(r"\b(the answer is|a|an|the)\b", " ", text)
    words = [_DIGIT_MAP.get(word, word) for word in text.split()]
    return " ".join(words)


def count_reasoning_records(total_prediction: str) -> int:
    return len(re.findall(r"Reasoning Record #\d+\s*:", str(total_prediction or "")))


def extract_last_reasoning_record(total_prediction: str) -> str:
    text = str(total_prediction or "")
    last_match_pos = text.rfind("Reasoning Record #")
    return "" if last_match_pos == -1 else text[last_match_pos:]


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or str(value).strip().lower() == "nan"


def _maybe_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return value


def _as_list(value: Any) -> list[Any]:
    value = _maybe_literal(value)
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _trajectory(prediction: dict[str, Any]) -> dict[str, Any]:
    trajectory = prediction.get("trajectory")
    return trajectory if isinstance(trajectory, dict) else {}


def _prediction_text(prediction: dict[str, Any]) -> str:
    trajectory = _trajectory(prediction)
    if trajectory.get("final_answer") not in (None, ""):
        return str(trajectory.get("final_answer") or "")
    return str(prediction.get("prediction") or prediction.get("answer") or "")


def _knowledge_text(prediction: dict[str, Any]) -> str:
    trajectory = _trajectory(prediction)
    if trajectory.get("all_knowledge") not in (None, ""):
        return str(trajectory.get("all_knowledge") or "")
    if prediction.get("knowledge") not in (None, ""):
        return str(prediction.get("knowledge") or "")

    parts: list[str] = []
    records = trajectory.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            for passage in record.get("text_results") or []:
                if not isinstance(passage, dict):
                    continue
                title = str(passage.get("title") or "").strip()
                text = str(passage.get("text") or "").strip()
                entry = f"{title}\n{text}" if title and text and title != text else text or title
                if entry:
                    parts.append(entry)
            for pair in record.get("image_results") or []:
                if not isinstance(pair, dict):
                    continue
                caption = str(pair.get("caption") or "").strip()
                if caption:
                    parts.append(caption)
    return "\n\n".join(parts)


def _text_knowledge_from_record(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for passage in record.get("text_results") or []:
        if not isinstance(passage, dict):
            continue
        title = str(passage.get("title") or "").strip()
        text = str(passage.get("text") or "").strip()
        entry = f"{title}\n{text}" if title and text and title != text else text or title
        if entry:
            parts.append(entry)
    return "\n\n".join(parts)


def _image_knowledge_from_record(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for pair in record.get("image_results") or []:
        if not isinstance(pair, dict):
            continue
        caption = str(pair.get("caption") or "").strip()
        if caption:
            parts.append(caption)
    return "\n\n".join(parts)


def _legacy_knowledge_runs(knowledge: str) -> list[tuple[str, str]]:
    blocks = re.split(r"(?=^Passage Title:|^Passage:)", str(knowledge or ""), flags=re.M)
    runs: list[tuple[str, str]] = []
    for block in blocks:
        chunk = block.strip()
        if not chunk:
            continue
        if chunk.startswith("Passage Title:"):
            runs.append(("text", chunk))
        elif chunk.startswith("Passage:"):
            runs.append(("image", chunk))
    return runs


def _chunk_legacy_runs(runs: list[tuple[str, str]], steps: int) -> list[list[tuple[str, str]]]:
    if not runs:
        return [[] for _ in range(max(steps, 1))]
    steps = max(steps, 1)
    if len(runs) % steps == 0:
        size = len(runs) // steps
        return [runs[i * size : (i + 1) * size] for i in range(steps)]

    chunks: list[list[tuple[str, str]]] = []
    for i in range(steps):
        start = round(i * len(runs) / steps)
        end = round((i + 1) * len(runs) / steps)
        chunks.append(runs[start:end])
    return chunks


def _iterative_knowledge_chunks(prediction: dict[str, Any]) -> list[dict[str, str]]:
    records = _trajectory(prediction).get("records")
    if isinstance(records, list) and any(
        isinstance(record, dict) and (record.get("text_results") or record.get("image_results"))
        for record in records
    ):
        chunks: list[dict[str, str]] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            chunks.append(
                {
                    "text": _text_knowledge_from_record(record),
                    "image": _image_knowledge_from_record(record),
                }
            )
        return chunks

    if isinstance(records, list) and records:
        runs = _legacy_knowledge_runs(_knowledge_text(prediction))
        grouped_runs = _chunk_legacy_runs(runs, len(records))
        chunks = []
        for run_group in grouped_runs:
            text_parts = [chunk for kind, chunk in run_group if kind == "text"]
            image_parts = [chunk for kind, chunk in run_group if kind == "image"]
            chunks.append(
                {
                    "text": "\n\n".join(text_parts),
                    "image": "\n\n".join(image_parts),
                }
            )
        return chunks

    knowledge = _knowledge_text(prediction)
    return [{"text": knowledge, "image": ""}] if knowledge else []


def _all_reasoning_text(prediction: dict[str, Any]) -> str:
    trajectory = _trajectory(prediction)
    if trajectory.get("all_reasoning") not in (None, ""):
        return str(trajectory.get("all_reasoning") or "")
    return str(prediction.get("total_pred") or "")


def count_reasoning_records_from_prediction(prediction: dict[str, Any]) -> int:
    records = _trajectory(prediction).get("records")
    if isinstance(records, list):
        return len([record for record in records if isinstance(record, dict) and record.get("reasoning")])
    return count_reasoning_records(_all_reasoning_text(prediction))


def extract_last_reasoning_record_from_prediction(prediction: dict[str, Any]) -> str:
    records = _trajectory(prediction).get("records")
    if isinstance(records, list):
        for record in reversed(records):
            if isinstance(record, dict) and record.get("reasoning"):
                return str(record.get("reasoning") or "")
    return extract_last_reasoning_record(_all_reasoning_text(prediction))


def _reference_values(prediction: dict[str, Any]) -> list[Any]:
    answer_eval = prediction.get("answer_eval")
    if not isinstance(answer_eval, bool) and not _is_missing(answer_eval):
        references = _as_list(answer_eval)
        if references:
            return references

    gold_answer = prediction.get("gold_answer")
    if not _is_missing(gold_answer):
        references = _as_list(gold_answer)
        if references:
            return references

    answer = prediction.get("answer")
    if not _is_missing(answer):
        references = _as_list(answer)
        if references:
            return references

    input_row = prediction.get("input")
    if isinstance(input_row, dict):
        answer_eval = input_row.get("answer_eval")
        if not isinstance(answer_eval, bool) and not _is_missing(answer_eval):
            references = _as_list(answer_eval)
            if references:
                return references
        answer = input_row.get("answer")
        if not _is_missing(answer):
            references = _as_list(answer)
            if references:
                return references

    return []


def _prediction_value(prediction: dict[str, Any], key: str) -> Any:
    if key in prediction:
        return prediction.get(key)
    input_row = prediction.get("input")
    if isinstance(input_row, dict):
        return input_row.get(key)
    return None


def _range_match(range_spec: dict[str, Any], prediction_text: str) -> bool:
    bounds = range_spec.get("range")
    if not isinstance(bounds, list) or len(bounds) != 2:
        return False
    try:
        lower = float(bounds[0])
        upper = float(bounds[1])
    except (TypeError, ValueError):
        return False
    for raw_number in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", prediction_text):
        try:
            number = float(raw_number)
        except ValueError:
            continue
        if lower <= number <= upper:
            return True
    return False


def _text_match(reference: Any, prediction_text: str) -> bool:
    if isinstance(reference, dict):
        return _range_match(reference, prediction_text)
    if reference is None:
        return False

    reference_text = str(reference).strip()
    if not reference_text:
        return False

    normalized_prediction = preprocess_answer(prediction_text)
    if "&&" in reference_text:
        sub_answers = [part for part in reference_text.split("&&") if part.strip()]
        if not sub_answers:
            return False
        hits = sum(preprocess_answer(part) in normalized_prediction for part in sub_answers)
        return hits / len(sub_answers) > 0.5

    if "|" in reference_text:
        return all(
            preprocess_answer(part) in normalized_prediction
            for part in reference_text.split("|")
            if part.strip()
        )

    normalized_reference = preprocess_answer(reference_text)
    if not normalized_reference:
        return False
    return normalized_reference in normalized_prediction


def _numeric_tolerance_result(answer: Any, prediction_text: str) -> bool | None:
    try:
        answer_value = float(answer)
    except (TypeError, ValueError):
        return None

    for raw_number in re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", prediction_text):
        try:
            number = float(raw_number)
        except ValueError:
            continue
        if abs(number - answer_value) <= 0.1 * answer_value:
            return True
    return False


def _answer_fields(prediction: dict[str, Any]) -> tuple[Any, Any, Any]:
    label = _prediction_value(prediction, "label")
    answer = _maybe_literal(_prediction_value(prediction, "answer"))
    ans_eval = _maybe_literal(_prediction_value(prediction, "answer_eval"))

    if _is_missing(answer):
        answer = _maybe_literal(_prediction_value(prediction, "gold_answer"))
    if _is_missing(ans_eval):
        ans_eval = ""
    return label, answer, ans_eval


def _cem_match(prediction: dict[str, Any]) -> bool:
    if isinstance(prediction.get("answer_eval"), bool):
        return bool(prediction["answer_eval"])

    prediction_text = _prediction_text(prediction)
    prediction_lower = prediction_text.lower()
    label, answer, ans_eval = _answer_fields(prediction)
    is_correct = False

    if label not in (None, ""):
        label_text = str(label)
        if label_text == prediction_text[: len(label_text)] or f"{label_text}:" in prediction_text:
            is_correct = True

    if isinstance(ans_eval, dict) and "range" in ans_eval and _range_match(ans_eval, prediction_text):
        is_correct = True

    if isinstance(ans_eval, list):
        for ans in ans_eval:
            if str(ans).lower() in prediction_lower:
                is_correct = True

    if isinstance(answer, list):
        for ans in answer:
            if isinstance(ans, dict):
                continue
            if isinstance(ans, str) and "&&" in ans:
                sub_answers = [part for part in ans.split("&&") if part.strip()]
                if sub_answers:
                    hits = sum(preprocess_answer(part) in prediction_lower for part in sub_answers)
                    if hits / len(sub_answers) > 0.5:
                        return True

            if str(ans).lower() in prediction_lower:
                is_correct = True

            if ans_eval == "":
                numeric_match = _numeric_tolerance_result(ans, prediction_text)
                if numeric_match is not None:
                    is_correct = numeric_match
        return is_correct

    if isinstance(answer, str) and answer:
        if answer.lower() in prediction_lower:
            is_correct = True

    return is_correct


def evaluate_cem_accuracy(predictions: list[dict[str, Any]]) -> tuple[float, list[bool]]:
    flags = [_cem_match(pred) for pred in predictions]
    return _score(flags), flags


def initialize_bem_model_and_transformers() -> None:
    """Initializes the strict TensorFlow Hub BEM tokenizer and model."""
    global bem_tokenizer, bem_model, bem_cls_id, bem_sep_id
    if (
        bem_tokenizer is not None
        and bem_model is not None
        and bem_cls_id is not None
        and bem_sep_id is not None
    ):
        return

    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text

    print("Initializing BEM model and tokenizer from TensorFlow Hub...")
    vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=_BEM_VOCAB_PATH,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        ),
        num_oov_buckets=1,
    )
    bem_cls_id, bem_sep_id = vocab_table.lookup(tf.convert_to_tensor(["[CLS]", "[SEP]"]))
    bem_tokenizer = text.BertTokenizer(
        vocab_lookup_table=vocab_table,
        token_out_type=tf.int64,
        preserve_unused_token=True,
        lower_case=True,
    )
    bem_model = hub.load(_BEM_MODEL_PATH)
    print("BEM model and tokenizer initialized successfully.")


def _squeeze_token_array(value: Any) -> Any:
    import numpy as np

    array = value.numpy()
    if array.ndim > 1:
        array = np.squeeze(array)
    return array


def _flatten_token_array(value: Any) -> Any:
    import numpy as np

    return np.asarray(value).reshape(-1)


def run_bem_evaluation(question: str, reference: str, candidate: str, *, verbose: bool = False) -> bool:
    """Run strict TensorFlow BEM answer-equivalence checking for one pair."""
    global bem_tokenizer, bem_model, bem_cls_id, bem_sep_id
    if not all([bem_tokenizer, bem_model, bem_cls_id is not None, bem_sep_id is not None]):
        raise RuntimeError("BEM components are not initialized.")

    import scipy.special
    import tensorflow_text as text

    if "<answer>" in candidate:
        candidate = (
            candidate.replace("<answer>", "")
            .replace("</answer>", "")
            .replace("<reason>", "")
            .replace("</reason>", "")
            .replace("<think>", "")
            .replace("</think>", "")
        )

    question_tokens = bem_tokenizer.tokenize(question).merge_dims(1, 2)
    reference_tokens = bem_tokenizer.tokenize(str(reference)).merge_dims(1, 2)
    candidate_tokens = bem_tokenizer.tokenize(candidate).merge_dims(1, 2)
    input_ids, segment_ids = text.combine_segments(
        (candidate_tokens, reference_tokens, question_tokens), bem_cls_id, bem_sep_id
    )

    def pad(array: np.ndarray, length: int = _BEM_MAX_TOKENS) -> np.ndarray:
        current_length = array.shape[-1]
        if current_length >= length:
            return array[-length:]
        return np.append(array, np.zeros(length - current_length, np.int32))

    input_ids_array = input_ids.numpy()
    segment_ids_array = segment_ids.numpy()
    input_ids_array = _flatten_token_array(input_ids_array)
    segment_ids_array = _flatten_token_array(segment_ids_array)
    if verbose and len(input_ids_array) > _BEM_MAX_TOKENS:
        print(f"[BEM] truncated combined sequence {len(input_ids_array)}->{_BEM_MAX_TOKENS}")

    inputs = {
        "input_ids": np.expand_dims(pad(input_ids_array), 0),
        "segment_ids": np.expand_dims(pad(segment_ids_array), 0),
    }
    raw_outputs = bem_model(inputs)
    bem_score = scipy.special.softmax(np.squeeze(raw_outputs))[1]
    return bool(bem_score >= 0.5)


def evaluate_bem_accuracy(
    predictions: list[dict[str, Any]],
    *,
    verbose: bool = False,
) -> tuple[float, list[bool]]:
    initialize_bem_model_and_transformers()
    flags: list[bool] = []
    for pred in predictions:
        question = str(pred.get("question") or "")
        prediction_text = _prediction_text(pred)
        references = [str(ref) for ref in _reference_values(pred) if not isinstance(ref, dict)]
        if not references:
            flags.append(False)
            continue

        joined_references = ",".join(references)
        if verbose:
            is_correct = run_bem_evaluation(question, joined_references, prediction_text, verbose=True)
        else:
            is_correct = run_bem_evaluation(question, joined_references, prediction_text)
        if not is_correct:
            if verbose:
                is_correct = any(
                    run_bem_evaluation(question, reference, prediction_text, verbose=True)
                    for reference in references
                )
            else:
                is_correct = any(
                    run_bem_evaluation(question, reference, prediction_text)
                    for reference in references
                )
        flags.append(is_correct)
    return _score(flags), flags


def evaluate_accuracy(
    predictions: list[dict[str, Any]],
    args: argparse.Namespace | None = None,
) -> tuple[float, list[bool]]:
    if args is not None and getattr(args, "bem", False):
        return evaluate_bem_accuracy(predictions, verbose=getattr(args, "verbose", False))
    return evaluate_cem_accuracy(predictions)


def _target_values_for_recall(prediction: dict[str, Any]) -> list[Any]:
    entity_text = prediction.get("entity_text")
    if not _is_missing(entity_text):
        return _as_list(entity_text)
    return _reference_values(prediction)


def _strip_accents(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )


def _knowledge_match(target: Any, knowledge: str) -> bool:
    if isinstance(target, dict):
        return False
    target_text = str(target).strip()
    if not target_text:
        return False
    normalized_knowledge = preprocess_answer(knowledge)
    if "|" in target_text:
        return all(
            preprocess_answer(part) in normalized_knowledge
            for part in target_text.split("|")
            if part.strip()
        )
    return preprocess_answer(target_text) in normalized_knowledge


def _recall_match(prediction: dict[str, Any], knowledge: str) -> bool:
    entity_text = _maybe_literal(prediction.get("entity_text"))
    if isinstance(entity_text, list):
        for entity in entity_text:
            entity_text_value = str(entity)
            if "|" in entity_text_value:
                names = [name for name in entity_text_value.split("|") if name.strip()]
                if names and all(_knowledge_match(name, knowledge) for name in names):
                    return True
            elif _knowledge_match(entity_text_value, knowledge) or _knowledge_match(_strip_accents(entity_text_value), knowledge):
                return True
        return False

    if not _is_missing(entity_text):
        return _knowledge_match(entity_text, knowledge)

    if any(_knowledge_match(target, knowledge) for target in _reference_values(prediction)):
        return True

    answer = prediction.get("answer")
    return any(_knowledge_match(target, knowledge) for target in _as_list(answer))


def _normalized_recall_target_groups(prediction: dict[str, Any]) -> list[list[str]]:
    entity_text = _maybe_literal(prediction.get("entity_text"))
    groups: list[list[str]] = []

    def add_value(value: Any, include_stripped: bool = False) -> None:
        text = str(value).strip()
        if not text:
            return
        if "|" in text:
            parts = [preprocess_answer(part) for part in text.split("|") if part.strip()]
            if parts:
                groups.append(parts)
            return
        normalized = preprocess_answer(text)
        if normalized:
            groups.append([normalized])
        if include_stripped:
            stripped = preprocess_answer(_strip_accents(text))
            if stripped and stripped != normalized:
                groups.append([stripped])

    if isinstance(entity_text, list):
        for entity in entity_text:
            add_value(entity, include_stripped=True)
        return groups

    if not _is_missing(entity_text):
        add_value(entity_text)
        return groups

    for target in _reference_values(prediction):
        if isinstance(target, dict):
            continue
        add_value(target)
    if groups:
        return groups

    for target in _as_list(prediction.get("answer")):
        add_value(target)
    return groups


def _normalized_recall_match(target_groups: list[list[str]], normalized_knowledge: str) -> bool:
    if not normalized_knowledge:
        return False
    for group in target_groups:
        if group and all(target in normalized_knowledge for target in group):
            return True
    return False


def evaluate_recall(
    predictions: list[dict[str, Any]],
    args: argparse.Namespace | None = None,
) -> tuple[float, list[bool]]:
    flags: list[bool] = []
    for pred in predictions:
        knowledge = _knowledge_text(pred)
        flags.append(_recall_match(pred, knowledge) if knowledge else False)
    return _score(flags), flags


def iterative_recall_breakdown(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_steps = 0
    chunked_predictions: list[list[dict[str, str]]] = []
    target_groups: list[list[list[str]]] = []
    for pred in predictions:
        chunks = _iterative_knowledge_chunks(pred)
        chunked_predictions.append(chunks)
        target_groups.append(_normalized_recall_target_groups(pred))
        max_steps = max(max_steps, len(chunks))

    summary: list[dict[str, Any]] = []
    cumulative_text_hits = [False] * len(predictions)
    cumulative_image_hits = [False] * len(predictions)
    cumulative_combined_hits = [False] * len(predictions)
    cumulative_text_knowledge = [""] * len(predictions)
    cumulative_image_knowledge = [""] * len(predictions)

    for step in range(max_steps):
        new_text_hits = 0
        new_image_hits = 0
        new_combined_hits = 0
        for idx, pred in enumerate(predictions):
            chunks = chunked_predictions[idx]
            if step < len(chunks):
                text_chunk = preprocess_answer(chunks[step].get("text") or "")
                image_chunk = preprocess_answer(chunks[step].get("image") or "")
                if text_chunk:
                    cumulative_text_knowledge[idx] = (
                        f"{cumulative_text_knowledge[idx]} {text_chunk}".strip()
                        if cumulative_text_knowledge[idx]
                        else text_chunk
                    )
                if image_chunk:
                    cumulative_image_knowledge[idx] = (
                        f"{cumulative_image_knowledge[idx]} {image_chunk}".strip()
                        if cumulative_image_knowledge[idx]
                        else image_chunk
                    )

            text_hit = (
                _normalized_recall_match(target_groups[idx], cumulative_text_knowledge[idx])
                if cumulative_text_knowledge[idx]
                else False
            )
            image_hit = (
                _normalized_recall_match(target_groups[idx], cumulative_image_knowledge[idx])
                if cumulative_image_knowledge[idx]
                else False
            )
            combined_knowledge = " ".join(
                part for part in (cumulative_text_knowledge[idx], cumulative_image_knowledge[idx]) if part
            )
            combined_hit = (
                _normalized_recall_match(target_groups[idx], combined_knowledge)
                if combined_knowledge
                else False
            )

            if text_hit and not cumulative_text_hits[idx]:
                new_text_hits += 1
            if image_hit and not cumulative_image_hits[idx]:
                new_image_hits += 1
            if combined_hit and not cumulative_combined_hits[idx]:
                new_combined_hits += 1

            cumulative_text_hits[idx] = text_hit
            cumulative_image_hits[idx] = image_hit
            cumulative_combined_hits[idx] = combined_hit

        total = len(predictions)
        summary.append(
            {
                "step": step,
                "text_hits": sum(cumulative_text_hits),
                "image_hits": sum(cumulative_image_hits),
                "combined_hits": sum(cumulative_combined_hits),
                "text_recall": _score(cumulative_text_hits),
                "image_recall": _score(cumulative_image_hits),
                "combined_recall": _score(cumulative_combined_hits),
                "new_text_hits": new_text_hits,
                "new_image_hits": new_image_hits,
                "new_combined_hits": new_combined_hits,
                "new_text_recall": new_text_hits / total if total else 0.0,
                "new_image_recall": new_image_hits / total if total else 0.0,
                "new_combined_recall": new_combined_hits / total if total else 0.0,
            }
        )
    return summary


def _print_iterative_recall(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("Iterative recall: no stepwise knowledge available.")
        return
    print("Iterative Recall:")
    print("step\ttext\timage\tcombined\tnew_text\tnew_image\tnew_combined")
    for row in summary:
        print(
            f"{row['step']}\t"
            f"{row['text_recall']:.2%} ({row['text_hits']})\t"
            f"{row['image_recall']:.2%} ({row['image_hits']})\t"
            f"{row['combined_recall']:.2%} ({row['combined_hits']})\t"
            f"{row['new_text_recall']:.2%} ({row['new_text_hits']})\t"
            f"{row['new_image_recall']:.2%} ({row['new_image_hits']})\t"
            f"{row['new_combined_recall']:.2%} ({row['new_combined_hits']})"
        )


def _score(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def _print_verbose_failures(predictions: list[dict[str, Any]], flags: list[bool]) -> None:
    for pred, flag in zip(predictions, flags):
        if flag:
            continue
        print(
            "##Q: {question}\n"
            "##Image Path: {image_path}\n"
            "##Gold: {gold}\n"
            "##Pred: {prediction}\n"
            "##Last Reasoning Record:\n{record}\n".format(
                question=pred.get("question", "N/A"),
                image_path=pred.get("image_path", "N/A"),
                gold=pred.get("gold_answer", "N/A"),
                prediction=_prediction_text(pred),
                record=extract_last_reasoning_record_from_prediction(pred),
            )
        )


def main() -> int:
    args = parse_args()
    predictions = load_predictions(args)
    if args.max_samples is not None:
        predictions = predictions[: args.max_samples]
        print(f"Evaluating only the first {len(predictions)} samples")

    accuracy, accuracy_flags = evaluate_accuracy(predictions, args)
    recall, recall_flags = evaluate_recall(predictions, args)
    metric_name = "BEM Accuracy" if args.bem else "CEM Accuracy"

    total = len(predictions)
    print(f"{metric_name}: {accuracy:.2%} ({sum(accuracy_flags)}/{total})")
    print(f"Recall: {recall:.2%} ({sum(recall_flags)}/{total})")

    record_counts = [count_reasoning_records_from_prediction(pred) for pred in predictions]
    nonzero_counts = [count for count in record_counts if count > 0]
    if nonzero_counts:
        print(f"Average reasoning records: {sum(nonzero_counts) / len(nonzero_counts):.2f}")

    if args.iterative_recall:
        _print_iterative_recall(iterative_recall_breakdown(predictions))

    if args.verbose:
        _print_verbose_failures(predictions, accuracy_flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
