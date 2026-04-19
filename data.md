# Dataset Preparation

PMSR expects evaluation data in JSONL format. Each line should contain the question, the local image path, and any available answer or entity metadata.

```json
{"question": "What country is this landmark located in?", "image_path": "/path/to/image.jpg", "answer": "France", "entity_text": "Eiffel Tower"}
```

The repository ignores local datasets and outputs by default. Keep downloaded benchmark files under `data/` or pass an absolute path with `--data`.

For broader and more up-to-date textual knowledge, we recommend trying the latest English Wikipedia monthly split from [wikipedia-monthly](https://huggingface.co/datasets/omarkamali/wikipedia-monthly). This can be used to expand or replace the text-only Wikipedia KB after converting the English split into PMSR metadata format and encoding it with the text embedding server. See [KB.md](KB.md) for text KB indexing instructions.

## InfoSeek

For PMSR evaluation, we use the **M2KR split of InfoSeek validation**. Download it from the [Multi-task Multi-modal Knowledge Retrieval Benchmark (M2KR)](https://huggingface.co/datasets/BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR) and use the `Infoseek_data` subset.

For the original InfoSeek resources, download the annotations and knowledge files from the [InfoSeek repository](https://github.com/open-vision-language/infoseek). The repository provides the benchmark splits, entity mappings, and the 6M Wikipedia text information used for knowledge-base construction.

After downloading, convert the target split into PMSR JSONL format:

```text
data/InfoSeek_val.jsonl
```

Each record should include at least:

- `question`: input question.
- `image_path`: local path to the corresponding image.
- `answer`: gold answer when available.
- `entity_text`, `wikidata_id`, `wikipedia_title`, or other metadata if available.

## E-VQA

Prepare the Encyclopedic-VQA split in the same JSONL format. The default development setup uses:

```text
data/EVQA_test.jsonl
```

## LiveVQA

Download [LiveVQA-Research-Preview](https://huggingface.co/datasets/ONE-Lab/LiveVQA-Research-Preview) from Hugging Face. The dataset includes an `image/` directory, `qa.json`, and `qa_detailed.json`. Access may require accepting the dataset terms on Hugging Face.

Convert `qa.json` into PMSR JSONL format:

- `question`: use `query`.
- `image_path`: use the local path under the downloaded `image/` directory.
- `gold_answer`: use `gt_answer`.
- `answer` and `answer_eval`: use `gt_answer`.
- Preserve `sample_id`, and optionally merge `topic` or `context` from `qa_detailed.json`.

Recommended output path:

```text
data/LiveVQA_test.jsonl
```

Prepare the preview split with:

```bash
python scripts/process_livevqa_preview.py \
  --output-jsonl data/LiveVQA_test.jsonl \
  --image-root data/processed/livevqa/images
```

If you already downloaded the gated Hugging Face repository, pass the local snapshot path:

```bash
python scripts/process_livevqa_preview.py \
  --snapshot-dir /path/to/LiveVQA-Research-Preview \
  --output-jsonl data/LiveVQA_test.jsonl \
  --image-root data/processed/livevqa/images
```

## FVQA

Download [FVQA](https://huggingface.co/datasets/lmms-lab/FVQA/) from Hugging Face. The dataset is stored in Parquet format with train and test splits. Use the test split for evaluation.

FVQA stores images as bytes in the Parquet file. Export each image to a local image directory, then write PMSR JSONL records:

- `question`: use the user text from `prompt`.
- `image_path`: use the exported local image path.
- `gold_answer`: use `reward_model.ground_truth`.
- `answer` and `answer_eval`: use `reward_model.candidate_answers` when available, otherwise fall back to `reward_model.ground_truth`.
- Preserve `data_id`, `category`, `candidate_answers`, and `image_urls` when available.

Recommended output path:

```text
data/fvqa_test.jsonl
```

Prepare the test split with:

```bash
python scripts/process_fvqa_test.py \
  --split test \
  --output-jsonl data/fvqa_test.jsonl \
  --image-root data/processed/fvqa/images
```

The script uses `huggingface_hub.snapshot_download` with split-specific patterns, so it downloads `fvqa_test.parquet` rather than the cached search-result pickle files.
By default, the Hugging Face cache for this script is kept under `data/processed/fvqa/hf_cache`; pass `--cache-dir` to place it elsewhere.

## InfoSeek Human

For InfoSeek Human, use the MMSearch-R1 InfoSeek subset from [multimodal-search-r1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1/tree/main/mmsearch_r1):

```text
multimodal-search-r1/mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet
```

This Parquet file follows the MMSearch-R1 style format with embedded image bytes, prompt messages, and reward-model answers. Export images to disk and convert each row into PMSR JSONL:

- `question`: use the user text from `prompt`.
- `image_path`: use the exported local image path.
- `gold_answer`: use `reward_model.ground_truth`.
- `answer` and `answer_eval`: use `reward_model.candidate_answers` when available, otherwise fall back to `reward_model.ground_truth`.
- Preserve `data_id`, `image_id`, `candidate_answers`, and `data_source`.

Recommended output path:

```text
data/InfoSeek_human_2k.jsonl
```

Prepare the subset with:

```bash
python scripts/process_infoseek_human_subset.py \
  --output-jsonl data/InfoSeek_human_2k.jsonl \
  --image-root data/processed/infoseek_human/images
```

If the Parquet file is already downloaded, pass it directly:

```bash
python scripts/process_infoseek_human_subset.py \
  --parquet-path /path/to/mmsearch_r1_infoseek_sub_2k.parquet \
  --output-jsonl data/InfoSeek_human_2k.jsonl \
  --image-root data/processed/infoseek_human/images
```

## MMSearch

Download [MMSearch](https://huggingface.co/datasets/CaraJ/MMSearch) from Hugging Face. For end-to-end PMSR evaluation, use the `end2end` subset. The `rerank` and `summarization` subsets are intended for task-specific evaluation.

Convert the `end2end` split into PMSR JSONL:

- `question`: use `query`.
- `image_path`: export `query_image` when present.
- `gold_answer`: use `gt_answer`.
- `answer` and `answer_eval`: use `gt_answer` plus `alternative_gt_answers`.
- Preserve `sample_id`, `gt_requery`, `area`, `subfield`, `timestamp`, and alternative answers.
- Keep only rows where `query_image` is not empty.

Recommended output path:

```text
data/MMSearch_end2end.jsonl
```

Prepare the end-to-end split with:

```bash
python scripts/process_mmsearch_end2end.py \
  --output-jsonl data/MMSearch_end2end.jsonl \
  --image-root data/processed/mmsearch/end2end/images
```

If `end2end.parquet` is already downloaded, pass it directly:

```bash
python scripts/process_mmsearch_end2end.py \
  --parquet-path /path/to/end2end.parquet \
  --output-jsonl data/MMSearch_end2end.jsonl \
  --image-root data/processed/mmsearch/end2end/images
```
