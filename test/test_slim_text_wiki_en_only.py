from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts import slim_text_wiki_en_only as slim


class SlimTextWikiEnOnlyTest(unittest.TestCase):
    def test_suffix_path_adds_en_before_extension(self) -> None:
        self.assertEqual(
            slim.suffix_path(Path("/tmp/wiki18_100w.jsonl")),
            Path("/tmp/wiki18_100w_en.jsonl"),
        )
        self.assertEqual(
            slim.suffix_path(Path("/tmp/wiki18_100w_e5_flat.index")),
            Path("/tmp/wiki18_100w_e5_flat_en.index"),
        )

    def test_is_english_text_uses_row_level_threshold(self) -> None:
        self.assertTrue(slim.is_english_text('"Pavia Cathedral"\nPavia Cathedral was begun in 1488.'))
        self.assertTrue(slim.is_english_text("English text with an en dash – and São Paulo."))
        self.assertFalse(slim.is_english_text("한글 문장입니다. 한국어 위키 문서입니다."))

    def test_filter_metadata_writes_only_english_rows_and_kept_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source = tmp / "wiki.jsonl"
            output = tmp / "wiki_en.jsonl"
            ids = tmp / "wiki_en.ids"
            rows = [
                {"id": "0", "contents": '"Alpha"\nThis is an English row.'},
                {"id": "1", "contents": "한국어 문서입니다."},
                {"id": "2", "contents": '"Beta"\nAnother English row.'},
            ]
            source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            stats = slim.filter_metadata(source, output, ids, threshold=0.90)

            self.assertEqual(stats.total_rows, 3)
            self.assertEqual(stats.kept_rows, 2)
            kept = [json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(kept, ["0", "2"])
            self.assertEqual(list(slim.iter_kept_ids(ids, batch_size=1)), [[0], [2]])


if __name__ == "__main__":
    unittest.main()
