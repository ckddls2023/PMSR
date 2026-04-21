from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.schemas import Evidence, Record, SearchResult, Trajectory
from eval.main import (
    _eval_answer,
    _record_id,
    build_config_from_args,
    build_output_path,
    maybe_merge_chunk_outputs,
    output_from_trajectory,
    slice_chunk,
)


class EvalMainTest(unittest.TestCase):
    def test_main_imports_metrics_from_metric_eval(self) -> None:
        import eval.main as eval_main
        import eval.metric_eval as metric_eval

        self.assertIs(eval_main.evaluate_accuracy, metric_eval.evaluate_accuracy)
        self.assertIs(eval_main.evaluate_recall, metric_eval.evaluate_recall)

    def test_build_config_strips_vllm_model_prefix(self) -> None:
        args = argparse.Namespace(
            model="vllm:Qwen/Qwen2.5-VL-7B-Instruct",
            api_base="http://127.0.0.1:8000/v1/chat/completions",
            api_key="",
            max_tokens=128,
            temperature=0.0,
            timeout=10,
            retry=0,
            text_kb="",
            text_metadata="",
            text_embed_api_base="",
            pmsr_kb="",
            pmsr_metadata="",
            mllm_kb="",
            mllm_metadata="",
            mllm_embed_api_base="",
            mllm_model="",
            image_embed_api_base="",
            pmsr_text_embed_api_base="",
            pmsr_fusion="concat",
            return_images=True,
            itercount=1,
            topk=2,
            threshold=0.9,
            verbose=False,
        )

        config = build_config_from_args(args)

        self.assertEqual(config.model, "Qwen/Qwen2.5-VL-7B-Instruct")

    def test_output_path_uses_data_jsonl_stem_as_prefix(self) -> None:
        args = argparse.Namespace(
            data="data/InfoSeek_val.jsonl",
            output_dir="outputs",
            model="Qwen/Qwen3.5-9B",
            api_base="",
            api_key="",
            max_tokens=128,
            temperature=0.0,
            timeout=10,
            retry=0,
            text_kb="/tmp/text.index",
            text_metadata="/tmp/text.jsonl",
            text_embed_api_base="http://text",
            pmsr_kb="/tmp/pmsr.index",
            pmsr_metadata="/tmp/pmsr.csv",
            mllm_kb="",
            mllm_metadata="",
            mllm_embed_api_base="",
            mllm_model="",
            image_embed_api_base="http://image",
            pmsr_text_embed_api_base="http://qwen",
            pmsr_fusion="concat",
            return_images=True,
            itercount=3,
            topk=10,
            threshold=0.9,
            verbose=False,
        )
        config = build_config_from_args(args)

        output_path = build_output_path(args, config)

        self.assertTrue(config.return_images)
        self.assertEqual(output_path.name, "InfoSeek_val_Qwen3.5-9B_iter3_topk10_text_pmsr.jsonl")

    def test_output_path_includes_mllm_fusion_to_avoid_resuming_concat_outputs(self) -> None:
        args = argparse.Namespace(
            data="data/InfoSeek_val.jsonl",
            output_dir="outputs",
            model="Qwen/Qwen3.5-9B",
            api_base="",
            api_key="",
            max_tokens=128,
            temperature=0.0,
            timeout=10,
            retry=0,
            text_kb="/tmp/text.index",
            text_metadata="/tmp/text.jsonl",
            text_embed_api_base="http://text",
            text_model="",
            pmsr_kb="/tmp/pmsr.index",
            pmsr_metadata="/tmp/pmsr.csv",
            mllm_kb="/tmp/mllm.index",
            mllm_metadata="/tmp/mllm.csv",
            mllm_embed_api_base="http://mllm",
            mllm_model="Qwen/Qwen3-VL-Embedding-2B",
            image_embed_api_base="",
            pmsr_text_embed_api_base="",
            pmsr_fusion="mllm",
            return_images=True,
            itercount=3,
            topk=10,
            threshold=0.9,
            verbose=False,
        )
        config = build_config_from_args(args)

        output_path = build_output_path(args, config)

        self.assertEqual(output_path.name, "InfoSeek_val_Qwen3.5-9B_iter3_topk10_text_mllm.jsonl")

    def test_build_config_uses_args_return_images(self) -> None:
        args = argparse.Namespace(
            model="Qwen/Qwen3.5-9B",
            api_base="",
            api_key="",
            max_tokens=128,
            temperature=0.0,
            timeout=10,
            retry=0,
            text_kb="",
            text_metadata="",
            text_embed_api_base="",
            pmsr_kb="",
            pmsr_metadata="",
            mllm_kb="",
            mllm_metadata="",
            mllm_embed_api_base="",
            mllm_model="",
            image_embed_api_base="",
            pmsr_text_embed_api_base="",
            pmsr_fusion="concat",
            return_images=False,
            itercount=1,
            topk=2,
            threshold=0.9,
            verbose=False,
        )

        config = build_config_from_args(args)

        self.assertFalse(config.return_images)

    def test_build_config_reads_mllm_retrieval_args(self) -> None:
        args = argparse.Namespace(
            model="Qwen/Qwen3.5-9B",
            api_base="",
            api_key="",
            max_tokens=128,
            temperature=0.0,
            timeout=10,
            retry=0,
            text_kb="",
            text_metadata="",
            text_embed_api_base="",
            text_model="",
            pmsr_kb="/tmp/pmsr.index",
            pmsr_metadata="/tmp/pmsr.csv",
            mllm_kb="/tmp/mllm.index",
            mllm_metadata="/tmp/mllm.csv",
            mllm_embed_api_base="http://mllm",
            mllm_model="Qwen/Qwen3-VL-Embedding-2B",
            image_embed_api_base="",
            pmsr_text_embed_api_base="",
            pmsr_fusion="mllm",
            return_images=True,
            itercount=1,
            topk=2,
            threshold=0.9,
            verbose=False,
        )

        config = build_config_from_args(args)

        self.assertEqual(config.mllm_kb, "/tmp/mllm.index")
        self.assertEqual(config.mllm_metadata, "/tmp/mllm.csv")
        self.assertEqual(config.mllm_embed_api_base, "http://mllm")
        self.assertEqual(config.mllm_model, "Qwen/Qwen3-VL-Embedding-2B")

    def test_parser_defaults_return_images_to_true(self) -> None:
        from eval.main import build_parser

        args = build_parser().parse_args([])

        self.assertTrue(args.return_images)

    def test_parser_defaults_generation_sampling_parameters(self) -> None:
        from eval.main import build_parser

        args = build_parser().parse_args([])
        config = build_config_from_args(args)

        self.assertEqual(args.top_p, 0.8)
        self.assertEqual(args.top_k, 20)
        self.assertEqual(config.top_p, 0.8)
        self.assertEqual(config.top_k, 20)

    def test_parser_without_image_sets_return_images_to_false(self) -> None:
        from eval.main import build_parser

        args = build_parser().parse_args(["--without-image"])

        self.assertFalse(args.return_images)

    def test_parser_without_traj_query_sets_config_and_output_suffix(self) -> None:
        from eval.main import build_parser

        args = build_parser().parse_args(
            [
                "--data",
                "data/EVQA_test.jsonl",
                "--model",
                "qwen/qwen3-vl-8b-instruct",
                "--without-traj-query",
            ]
        )
        config = build_config_from_args(args)
        output_path = build_output_path(args, config)

        self.assertFalse(args.use_traj_query)
        self.assertFalse(config.use_traj_query)
        self.assertIn("EVQA_test_qwen3-vl-8b-instruct_iter3_topk10", output_path.name)
        self.assertTrue(output_path.name.endswith("_without_ask.jsonl"))

    def test_parser_chunk_id_adds_chunk_suffix_to_output(self) -> None:
        from eval.main import build_parser

        args = build_parser().parse_args(
            [
                "--data",
                "data/EVQA_test.jsonl",
                "--model",
                "Qwen/Qwen3.5-9B",
                "--chunk-id",
                "3",
            ]
        )
        config = build_config_from_args(args)
        output_path = build_output_path(args, config)

        self.assertEqual(args.chunk_id, 3)
        self.assertTrue(output_path.name.endswith("_chunk_3.jsonl"))

    def test_slice_chunk_uses_stable_modulo_partition(self) -> None:
        rows = [{"question_id": f"q{i}"} for i in range(12)]

        chunk = slice_chunk(rows, chunk_id=3, num_chunks=10)

        self.assertEqual([row["question_id"] for row in chunk], ["q3"])
        self.assertEqual(
            [row["question_id"] for row in slice_chunk(rows, chunk_id=1, num_chunks=10)],
            ["q1", "q11"],
        )

    def test_record_id_uses_dataset_image_ids_for_input_rows(self) -> None:
        row = {
            "dataset_image_ids": "2715027",
            "image_path": "/tmp/example.jpg",
        }

        self.assertEqual(_record_id(row), "2715027")

    def test_maybe_merge_chunk_outputs_merges_when_all_chunks_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            base_path = output_dir / "EVQA_test_Qwen3-VL-8B_iter4_topk10_text_pmsr.jsonl"
            expected_items = [{"question_id": f"q{i}", "image_path": f"/tmp/{i}.jpg"} for i in range(20)]

            for chunk_id in range(10):
                chunk_path = output_dir / f"EVQA_test_Qwen3-VL-8B_iter4_topk10_text_pmsr_chunk_{chunk_id}.jsonl"
                chunk_predictions = []
                for index, item in enumerate(expected_items):
                    if index % 10 != chunk_id:
                        continue
                    chunk_predictions.append(
                        {
                            "question_id": item["question_id"],
                            "image_path": item["image_path"],
                            "trajectory": {"final_answer": item["question_id"]},
                        }
                    )
                with chunk_path.open("w", encoding="utf-8") as handle:
                    for record in chunk_predictions:
                        handle.write(json.dumps(record) + "\n")

            merged_path = maybe_merge_chunk_outputs(base_path, expected_items)

            self.assertEqual(merged_path, base_path)
            self.assertTrue(base_path.exists())
            with base_path.open("r", encoding="utf-8") as handle:
                merged_rows = [json.loads(line) for line in handle]
            self.assertEqual([row["question_id"] for row in merged_rows], [f"q{i}" for i in range(20)])

    def test_maybe_merge_chunk_outputs_skips_until_all_chunks_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            base_path = output_dir / "EVQA_test_Qwen3-VL-8B_iter4_topk10_text_pmsr.jsonl"
            expected_items = [{"question_id": f"q{i}", "image_path": f"/tmp/{i}.jpg"} for i in range(10)]

            for chunk_id in range(9):
                chunk_path = output_dir / f"EVQA_test_Qwen3-VL-8B_iter4_topk10_text_pmsr_chunk_{chunk_id}.jsonl"
                chunk_path.write_text("", encoding="utf-8")

            merged_path = maybe_merge_chunk_outputs(base_path, expected_items)

            self.assertIsNone(merged_path)
            self.assertFalse(base_path.exists())

    def test_web_search_flag_overrides_text_kb_with_ollama_web_search(self) -> None:
        from eval.main import build_parser

        args = build_parser().parse_args(["--web-search"])
        args.model = "Qwen/Qwen3.5-9B"
        args.api_base = ""
        args.api_key = ""
        args.max_tokens = 128
        args.temperature = 0.0
        args.timeout = 10
        args.retry = 0
        args.pmsr_kb = ""
        args.pmsr_metadata = ""
        args.mllm_kb = ""
        args.mllm_metadata = ""
        args.mllm_embed_api_base = ""
        args.mllm_model = ""
        args.image_embed_api_base = ""
        args.pmsr_text_embed_api_base = ""
        args.pmsr_fusion = "concat"
        args.return_images = True
        args.itercount = 1
        args.topk = 2
        args.threshold = 0.9
        args.verbose = False

        config = build_config_from_args(args)

        self.assertEqual(config.text_kb, "https://ollama.com/api/web_search")
        self.assertEqual(config.text_metadata, "")
        self.assertEqual(config.text_embed_api_base, "")

    def test_similarity_embedding_prefers_mllm_over_text_embed_api_base(self) -> None:
        from eval.main import build_parser

        args = build_parser().parse_args(
            [
                "--text-embed-api-base",
                "http://e5",
                "--pmsr-text-embed-api-base",
                "http://qwen-text",
                "--mllm-embed-api-base",
                "http://mllm",
                "--mllm-model",
                "Qwen/Qwen3-VL-Embedding-2B",
            ]
        )

        config = build_config_from_args(args)

        self.assertEqual(config.text_embed_api_base, "http://e5")
        self.assertEqual(config.similarity_embed_api_base, "http://mllm")
        self.assertEqual(config.similarity_model, "Qwen/Qwen3-VL-Embedding-2B")
        self.assertEqual(config.similarity_embed_mode, "mllm")

    def test_similarity_embedding_falls_back_to_qwen_text_env_not_text_env(self) -> None:
        from unittest.mock import patch
        from eval.main import build_parser

        args = build_parser().parse_args([])
        with patch.dict(
            "os.environ",
            {
                "TEXT_EMBED_API_BASE": "http://e5",
                "QWEN_TEXT_EMBED_API_BASE": "http://qwen-text",
            },
            clear=True,
        ):
            config = build_config_from_args(args)

        self.assertEqual(config.text_embed_api_base, "http://e5")
        self.assertEqual(config.similarity_embed_api_base, "http://qwen-text")
        self.assertEqual(config.similarity_model, "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(config.similarity_embed_mode, "text")

    def test_output_saves_direct_trajectory_with_dataset_metadata(self) -> None:
        item = {
            "question_id": "fvqa_test_0",
            "image_id": "img0",
            "question": "Where is it?",
            "image_path": "/tmp/image.jpg",
            "gold_answer": "Las Vegas",
            "answer_eval": ["Las Vegas", "Vegas"],
            "entity_text": "Las Vegas",
        }
        traj = Trajectory(question=item["question"], image_path=item["image_path"], final_answer="Las Vegas, Nevada")
        traj.records.append(
            Record(
                step=0,
                local_query="Question: Where is it?\nA casino building.",
                global_query="Question: Where is it?\nA casino building.",
                text_results=[
                    SearchResult(Evidence(source="text", modality="text", title="Las Vegas", text="City in Nevada."))
                ],
                image_results=[
                    SearchResult(Evidence(source="pmsr", modality="image", image_path="/tmp/ref.jpg", caption="Las Vegas Strip."))
                ],
                reasoning="The image and retrieved evidence point to Las Vegas.",
                elapsed=1.25,
            )
        )

        output = output_from_trajectory(item, traj)

        self.assertEqual(output["question_id"], "fvqa_test_0")
        self.assertEqual(output["image_id"], "img0")
        self.assertEqual(output["gold_answer"], "Las Vegas")
        self.assertEqual(output["answer_eval"], ["Las Vegas", "Vegas"])
        self.assertNotIn("prediction", output)
        self.assertNotIn("knowledge", output)
        self.assertNotIn("total_pred", output)
        self.assertEqual(output["trajectory"]["final_answer"], "Las Vegas, Nevada")
        self.assertEqual(
            output["trajectory"]["records"][0]["reasoning"],
            "The image and retrieved evidence point to Las Vegas.",
        )
        self.assertIn("City in Nevada.", output["trajectory"]["all_knowledge"])
        self.assertIn("Las Vegas Strip.", output["trajectory"]["all_knowledge"])

    def test_eval_answer_supports_list_and_range_targets(self) -> None:
        self.assertTrue(_eval_answer("It is used for urinary tract infections.", {"answer": ["urinary tract infections"]}))
        self.assertTrue(_eval_answer("The value is about 42 meters.", {"answer_eval": {"range": [40, 45]}}))
        self.assertFalse(_eval_answer("The value is about 30 meters.", {"answer_eval": {"range": [40, 45]}}))

    def test_singular_schema_module_reexports_record_types(self) -> None:
        from agents import Record, Trajectory as SchemaTrajectory

        self.assertEqual(Record.__name__, "Record")
        self.assertIs(SchemaTrajectory, Trajectory)


if __name__ == "__main__":
    unittest.main()
