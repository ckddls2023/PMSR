from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.main_react import build_config_from_args, build_output_path, build_parser


class MainReactTest(unittest.TestCase):
    def test_react_config_does_not_enable_text_retrieval(self) -> None:
        args = build_parser().parse_args(
            [
                "--text-kb",
                "/tmp/ignored.index",
                "--text-embed-api-base",
                "http://ignored",
            ]
        )

        config = build_config_from_args(args)

        self.assertEqual(config.text_kb, "")
        self.assertEqual(config.text_metadata, "")
        self.assertEqual(config.text_embed_api_base, "")

    def test_react_config_sets_optional_backends_for_single_pmsr_tool(self) -> None:
        args = build_parser().parse_args(["--web-search", "--google-lens-search"])

        config = build_config_from_args(args)
        output_path = build_output_path(args, config)

        self.assertTrue(config.web_search)
        self.assertTrue(config.google_lens_search)
        self.assertIn("_web", output_path.name)
        self.assertIn("_google_lens", output_path.name)

    def test_parser_without_image_sets_return_images_to_false(self) -> None:
        args = build_parser().parse_args(["--without-image"])

        config = build_config_from_args(args)

        self.assertFalse(args.return_images)
        self.assertFalse(config.return_images)

    def test_react_config_supports_mllm_pmsr_search(self) -> None:
        args = build_parser().parse_args(
            [
                "--data",
                "data/InfoSeek_val.jsonl",
                "--model",
                "Qwen/Qwen3.5-9B",
                "--pmsr-fusion",
                "mllm",
                "--mllm-kb",
                "/tmp/mllm.index",
                "--mllm-metadata",
                "/tmp/mllm.csv",
                "--mllm-embed-api-base",
                "http://mllm",
                "--mllm-model",
                "Qwen/Qwen3-VL-Embedding-2B",
            ]
        )

        config = build_config_from_args(args)
        output_path = build_output_path(args, config)

        self.assertEqual(config.mllm_kb, "/tmp/mllm.index")
        self.assertEqual(config.mllm_metadata, "/tmp/mllm.csv")
        self.assertEqual(config.mllm_embed_api_base, "http://mllm")
        self.assertEqual(output_path.name, "InfoSeek_val_Qwen3.5-9B_react_topk10_mllm.jsonl")


if __name__ == "__main__":
    unittest.main()
