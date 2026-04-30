"""Evaluation utilities: string-match accuracy, retrieval recall, and BEM."""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# String-match accuracy (reads pre-computed answer_eval field)
# ---------------------------------------------------------------------------


def evaluate_accuracy(predictions: list[dict[str, Any]]) -> tuple[float, list[bool]]:
    """Return (accuracy, per-item flags) using the pre-computed answer_eval field."""
    results = [bool(p.get("answer_eval", False)) for p in predictions]
    return (sum(results) / len(results) if results else 0.0), results


# ---------------------------------------------------------------------------
# Retrieval recall
# ---------------------------------------------------------------------------


def evaluate_recall(predictions: list[dict[str, Any]]) -> tuple[float, list[bool]]:
    """Check whether entity_text (or gold_answer) appears in the retrieved knowledge."""
    results: list[bool] = []
    for p in predictions:
        knowledge = str(p.get("knowledge", "")).lower()
        entity = str(p.get("entity_text", "") or p.get("gold_answer", "")).lower().strip()
        if not entity:
            results.append(False)
            continue
        parts = [part.strip() for part in entity.split("|") if part.strip()]
        results.append(all(part in knowledge for part in parts))
    return (sum(results) / len(results) if results else 0.0), results


# ---------------------------------------------------------------------------
# BEM (answer-equivalence) evaluation
# ---------------------------------------------------------------------------

_bem_model = None
_bem_tokenizer = None
_bem_cls_id = None
_bem_sep_id = None

_VOCAB_PATH = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12/vocab.txt"
_MODEL_PATH = "https://tfhub.dev/google/answer_equivalence/bem/1"


def initialize_bem() -> None:
    """Load BEM tokenizer and model from TensorFlow Hub (idempotent)."""
    global _bem_model, _bem_tokenizer, _bem_cls_id, _bem_sep_id
    if _bem_model is not None:
        return
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text as text  # noqa: F401 — registers BERT ops

    print("Initializing BEM model from TensorFlow Hub...")
    vocab_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.TextFileInitializer(
            filename=_VOCAB_PATH,
            key_dtype=tf.string,
            key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64,
            value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        ),
        num_oov_buckets=1,
    )
    _bem_cls_id, _bem_sep_id = vocab_table.lookup(tf.convert_to_tensor(["[CLS]", "[SEP]"]))
    _bem_tokenizer = text.BertTokenizer(
        vocab_lookup_table=vocab_table,
        token_out_type=tf.int64,
        preserve_unused_token=True,
        lower_case=True,
    )
    _bem_model = hub.load(_MODEL_PATH)
    print("BEM model initialized.")


def _bem_score(question: str, reference: str, candidate: str) -> float:
    import numpy as np
    import scipy.special
    import tensorflow_text as text  # noqa: F401

    # Strip reasoning tags that some models emit
    for tag in ("<answer>", "</answer>", "<reason>", "</reason>", "<think>", "</think>"):
        candidate = candidate.replace(tag, "")

    q_tok = _bem_tokenizer.tokenize(question).merge_dims(1, 2)
    ref_tok = _bem_tokenizer.tokenize(reference).merge_dims(1, 2)
    cand_tok = _bem_tokenizer.tokenize(candidate).merge_dims(1, 2)
    input_ids, segment_ids = text.combine_segments(
        (cand_tok, ref_tok, q_tok), _bem_cls_id, _bem_sep_id
    )

    def _pad(a: "np.ndarray", length: int = 512) -> "np.ndarray":
        a = np.squeeze(a)
        return a[:length] if len(a) >= length else np.append(a, np.zeros(length - len(a), np.int32))

    inputs = {
        "input_ids": np.expand_dims(_pad(input_ids.numpy()), 0),
        "segment_ids": np.expand_dims(_pad(segment_ids.numpy()), 0),
    }
    raw = _bem_model(inputs)
    return float(scipy.special.softmax(np.squeeze(raw))[1])


def evaluate_bem(predictions: list[dict[str, Any]]) -> tuple[float, list[bool]]:
    """Evaluate accuracy using the BEM answer-equivalence model (threshold 0.5)."""
    initialize_bem()
    results: list[bool] = []
    for p in predictions:
        question = str(p.get("question", ""))
        gold = str(p.get("gold_answer", p.get("answer", ""))).replace("&&", ",")
        prediction = str(p.get("prediction", p.get("answer", "")))
        score = _bem_score(question, gold, prediction)
        results.append(score >= 0.5)
    return (sum(results) / len(results) if results else 0.0), results
