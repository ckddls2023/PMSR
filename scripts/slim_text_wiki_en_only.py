#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


@dataclass(frozen=True)
class FilterStats:
    total_rows: int
    kept_rows: int

    @property
    def dropped_rows(self) -> int:
        return self.total_rows - self.kept_rows


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def suffix_path(path: str | Path, suffix: str = "_en") -> Path:
    input_path = Path(path).expanduser()
    return input_path.with_name(f"{input_path.stem}{suffix}{input_path.suffix}")


def is_english_text(text: str, *, threshold: float = 0.90) -> bool:
    ascii_alpha = len(ASCII_ALPHA_RE.findall(text))
    non_ascii = len(NON_ASCII_RE.findall(text))
    signal = ascii_alpha + non_ascii
    if signal == 0:
        return False
    return ascii_alpha / signal >= threshold


def filter_metadata(
    metadata_path: str | Path,
    output_metadata_path: str | Path,
    kept_ids_path: str | Path,
    *,
    threshold: float = 0.90,
) -> FilterStats:
    input_path = Path(metadata_path).expanduser()
    output_path = Path(output_metadata_path).expanduser()
    ids_path = Path(kept_ids_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output, ids_path.open("wb") as kept_ids:
        for line in tqdm(source, desc="Filtering metadata", unit=" rows"):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = total_rows
            total_rows += 1
            contents = str(row.get("contents") or "")
            if is_english_text(contents, threshold=threshold):
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept_ids.write(struct.pack("<q", row_id))
                kept_rows += 1
    return FilterStats(total_rows=total_rows, kept_rows=kept_rows)


def iter_kept_ids(kept_ids_path: str | Path, *, batch_size: int) -> Iterable[list[int]]:
    ids_path = Path(kept_ids_path).expanduser()
    item_size = struct.calcsize("<q")
    with ids_path.open("rb") as handle:
        while True:
            data = handle.read(item_size * batch_size)
            if not data:
                break
            count = len(data) // item_size
            yield [item[0] for item in struct.iter_unpack("<q", data[: count * item_size])]


def _make_flat_index(dimension: int, metric_type: int):
    import faiss

    if metric_type == faiss.METRIC_INNER_PRODUCT:
        return faiss.IndexFlatIP(dimension)
    if metric_type == faiss.METRIC_L2:
        return faiss.IndexFlatL2(dimension)
    return faiss.IndexFlat(dimension, metric_type)


def write_filtered_faiss_index(
    source_index_path: str | Path,
    output_index_path: str | Path,
    kept_ids_path: str | Path,
    *,
    total_rows: int,
    kept_rows: int,
    batch_size: int = 8192,
) -> None:
    source_path = Path(source_index_path).expanduser()
    output_path = Path(output_index_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if total_rows == kept_rows:
        shutil.copy2(source_path, output_path)
        return

    import faiss
    import numpy as np

    source_index = faiss.read_index(str(source_path))
    if source_index.ntotal != total_rows:
        raise ValueError(
            f"FAISS index ntotal ({source_index.ntotal}) does not match metadata rows ({total_rows})."
        )

    output_index = _make_flat_index(source_index.d, source_index.metric_type)
    for ids in tqdm(iter_kept_ids(kept_ids_path, batch_size=batch_size), desc="Copying FAISS vectors"):
        id_array = np.asarray(ids, dtype="int64")
        vectors = source_index.reconstruct_batch(id_array)
        output_index.add(np.asarray(vectors, dtype="float32"))

    if output_index.ntotal != kept_rows:
        raise ValueError(f"Output index ntotal ({output_index.ntotal}) does not match kept rows ({kept_rows}).")
    faiss.write_index(output_index, str(output_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create _en text KB files from TEXT_KB and TEXT_METADATA.")
    parser.add_argument("--env-file", default=".env", help="Path to .env file.")
    parser.add_argument("--text-kb", default=None, help="Input FAISS index. Defaults to TEXT_KB from .env/environment.")
    parser.add_argument(
        "--text-metadata",
        default=None,
        help="Input metadata JSONL. Defaults to TEXT_METADATA from .env/environment.",
    )
    parser.add_argument("--output-kb", default=None, help="Output FAISS index. Defaults to TEXT_KB with _en suffix.")
    parser.add_argument(
        "--output-metadata",
        default=None,
        help="Output metadata JSONL. Defaults to TEXT_METADATA with _en suffix.",
    )
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=8192, help="FAISS vector copy batch size.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only write _en metadata and kept id sidecar; skip FAISS index creation.",
    )
    parser.add_argument(
        "--keep-ids-path",
        default=None,
        help="Optional kept-row id sidecar path. Defaults to output metadata path plus .ids.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    load_env_file(args.env_file)

    text_kb_value = args.text_kb or os.environ.get("TEXT_KB")
    text_metadata_value = args.text_metadata or os.environ.get("TEXT_METADATA")
    if not text_kb_value:
        raise SystemExit("TEXT_KB is required via --text-kb or .env/environment.")
    if not text_metadata_value:
        raise SystemExit("TEXT_METADATA is required via --text-metadata or .env/environment.")
    text_kb = Path(text_kb_value).expanduser()
    text_metadata = Path(text_metadata_value).expanduser()

    output_kb = Path(args.output_kb).expanduser() if args.output_kb else suffix_path(text_kb)
    output_metadata = Path(args.output_metadata).expanduser() if args.output_metadata else suffix_path(text_metadata)
    kept_ids_path = Path(args.keep_ids_path).expanduser() if args.keep_ids_path else output_metadata.with_suffix(
        output_metadata.suffix + ".ids"
    )

    stats = filter_metadata(
        text_metadata,
        output_metadata,
        kept_ids_path,
        threshold=args.threshold,
    )
    print(
        f"metadata: kept {stats.kept_rows}/{stats.total_rows} rows "
        f"({stats.kept_rows / stats.total_rows:.2%}); dropped {stats.dropped_rows}"
    )
    print(f"metadata output: {output_metadata}")
    print(f"kept ids: {kept_ids_path}")

    if args.metadata_only:
        return

    write_filtered_faiss_index(
        text_kb,
        output_kb,
        kept_ids_path,
        total_rows=stats.total_rows,
        kept_rows=stats.kept_rows,
        batch_size=args.batch_size,
    )
    print(f"index output: {output_kb}")


if __name__ == "__main__":
    main()
