<div align="center">

# Progressive Multimodal Search and Reasoning for Knowledge-Intensive Visual Question Answering

**ACL 2026 Oral**

[![arXiv](https://img.shields.io/badge/arXiv-2509.00798-b31b1b.svg)](https://arxiv.org/abs/2509.00798)
[![Paper](https://img.shields.io/badge/Paper-PDF-blue.svg)](https://arxiv.org/pdf/2509.00798)

**Changin Choi, Wonseok Lee, Jungmin Ko, Wonjong Rhee**

</div>

## Overview

This is the official repository for **PMSR: Progressive Multimodal Search and Reasoning**, a **multimodal iterative RAG** framework accepted as an **Oral paper at ACL 2026**.

Knowledge-intensive visual question answering requires models to connect image content with external knowledge. Existing multimodal RAG systems commonly rely on a single retrieval pass, which can miss necessary evidence and leave early reasoning errors uncorrected. PMSR addresses this limitation with multimodal iterative retrieval-augmented generation, progressively building a structured reasoning trajectory. At each step, the model generates dual-scope search queries from both the latest reasoning record and the accumulated trajectory, retrieves evidence from heterogeneous knowledge bases, and composes the evidence into a compact updated record.

## News

- **2026.04**: PMSR was accepted to **ACL 2026 as an Oral paper**.
- **2026.04**: The arXiv version was updated.
- Code, data processing scripts, and evaluation instructions will be released soon.

## Method

<p align="center">
  <img src="assets/method.png" width="95%" alt="PMSR method overview">
</p>

PMSR performs iterative search and reasoning through three core ideas:

- **Progressive reasoning trajectory**: the model maintains a sequence of compact reasoning records instead of making a single-pass prediction.
- **Dual-scope query generation**: each iteration searches with both a latest-record query and a trajectory-level query to balance local refinement and global context.
- **Heterogeneous knowledge retrieval**: PMSR retrieves from both multimodal and textual knowledge bases, then synthesizes evidence through compositional reasoning.
- **Adaptive termination**: the search process can stop when the reasoning trajectory saturates, reducing unnecessary iterations while preserving performance.

## Main Results

### Retrieval Performance

PMSR improves cumulative evidence recall across knowledge-intensive VQA benchmarks.

| Method | InfoSeek Recall | E-VQA Recall | OK-VQA PRR |
| --- | ---: | ---: | ---: |
| PMSR (Qwen3-VL-4B) | 93.9 | 64.3 | 92.1 |
| PMSR (Qwen3-VL-8B) | 94.6 | 67.3 | 97.1 |

### End-to-End Accuracy

| Method | InfoSeek M2KR | E-VQA Single-hop |
| --- | ---: | ---: |
| PMSR (Qwen3-VL-4B) | 38.3 | 40.9 |
| PMSR (Qwen3-VL-8B) | 41.5 | 46.4 |
| PMSR (Gemini-2.5-Flash) | 50.5 | 59.9 |

### Search-Oriented Multimodal Benchmarks

Using Qwen2.5-VL-7B as the backbone, PMSR also performs strongly on broader search-oriented benchmarks.

| Method | FVQA-test | InfoSeek Human | MMSearch | LiveVQA |
| --- | ---: | ---: | ---: | ---: |
| PMSR | 61.2 | 58.2 | 54.3 | 54.2 |

## Repository Structure

```text
PMSR/
+-- assets/
+-- agents/
+-- api/
+-- data/
+-- eval/
+-- outputs/
+-- scripts/
+-- search/
`-- README.md
```

## Dataset Preparation

PMSR expects evaluation data in JSONL format. Each line should contain the question, the local image path, and any available answer or entity metadata.

```json
{"question": "What country is this landmark located in?", "image_path": "/path/to/image.jpg", "answer": "France", "entity_text": "Eiffel Tower"}
```

The repository ignores local datasets and outputs by default. Keep downloaded benchmark files under `data/` or pass an absolute path with `--data`.

### InfoSeek

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

### E-VQA

Prepare the Encyclopedic-VQA split in the same JSONL format. The default development setup uses:

```text
data/EVQA_test.jsonl
```

### LiveVQA

Download [LiveVQA-Research-Preview](https://huggingface.co/datasets/ONE-Lab/LiveVQA-Research-Preview) from Hugging Face. The dataset includes an `image/` directory, `qa.json`, and `qa_detailed.json`. Access may require accepting the dataset terms on Hugging Face.

Convert `qa.json` into PMSR JSONL format:

- `question`: use `query`.
- `image_path`: use the local path under the downloaded `image/` directory.
- `answer`: use `gt_answer`.
- Preserve `sample_id`, and optionally merge `topic` or `context` from `qa_detailed.json`.

Recommended output path:

```text
data/LiveVQA_test.jsonl
```

### FVQA

Download [FVQA](https://huggingface.co/datasets/lmms-lab/FVQA/) from Hugging Face. The dataset is stored in Parquet format with train and test splits. Use the test split for evaluation.

FVQA stores images as bytes in the Parquet file. Export each image to a local image directory, then write PMSR JSONL records:

- `question`: use the user text from `prompt`.
- `image_path`: use the exported local image path.
- `answer`: use `reward_model.ground_truth`.
- Preserve `data_id`, `category`, `candidate_answers`, and `image_urls` when available.

Recommended output path:

```text
data/fvqa_test.jsonl
```

### InfoSeek Human

For InfoSeek Human, use the MMSearch-R1 InfoSeek subset from [multimodal-search-r1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1/tree/main/mmsearch_r1):

```text
multimodal-search-r1/mmsearch_r1/data/mmsearch_r1_infoseek_sub_2k.parquet
```

This Parquet file follows the MMSearch-R1 style format with embedded image bytes, prompt messages, and reward-model answers. Export images to disk and convert each row into PMSR JSONL:

- `question`: use the user text from `prompt`.
- `image_path`: use the exported local image path.
- `answer`: use `reward_model.ground_truth`.
- Preserve `data_id`, `image_id`, `candidate_answers`, and `data_source`.

Recommended output path:

```text
data/InfoSeek_human_2k.jsonl
```

### MMSearch

Download [MMSearch](https://huggingface.co/datasets/CaraJ/MMSearch) from Hugging Face. For end-to-end PMSR evaluation, use the `end2end` subset. The `rerank` and `summarization` subsets are intended for task-specific evaluation.

Convert the `end2end` split into PMSR JSONL:

- `question`: use the main visual question.
- `image_path`: use the exported or downloaded local image path.
- `answer`: use the gold answer field.
- Preserve category, source, date, query, and alternative answers when available. When process end2end split, select only rows that corresponding language is english.

Recommended output path:

```text
data/MMSearch_end2end.jsonl
```

### Run Evaluation

The evaluation entry point reads any prepared PMSR JSONL file directly:

```bash
python eval/main.py \
  --data data/EVQA_test.jsonl \
  --model Qwen/Qwen3.5-9B \
  --api_base http://<host>:8004/ \
  --itercount 3 \
  --topk 10
```

## Knowledge Base Preparation

PMSR retrieves from textual and multimodal knowledge bases. For the multimodal Wikipedia KB, we pair Wikipedia text entries with local Wikipedia entity images, encode the image and text fields with embedding API servers, concatenate the embeddings, and write a FAISS index plus metadata.

### Wikipedia Image-Text KB

1. Download the 6M Wikipedia text information from the [InfoSeek repository](https://github.com/open-vision-language/infoseek).

2. Download Wikipedia entity images from the [OVEN dataset](https://huggingface.co/datasets/ychenNLP/oven). This dataset is large and may require Hugging Face login or dataset access approval.

3. Arrange the image files so `scripts/create_wikipedia_jsonl.py` can find them:

```text
wikipedia_images_full/
+-- Q806/
|   `-- Q806563.jpg
+-- Q42/
|   `-- Q42.jpg
`-- ...
```

The script expects the following local filenames by default:

```text
Wiki6M_ver_1_0.jsonl
wikipedia_images_full/<first-four-wikidata-chars>/<wikidata_id>.jpg
```

4. Create an image-text JSONL file containing only Wikipedia rows with available local images:

```bash
python scripts/create_wikipedia_jsonl.py
```

This writes:

```text
Wiki6M_ver_1_0_updated.jsonl
```

5. Start the embedding servers. PMSR uses SigLIP2 for image embeddings and Qwen3 Embedding for text embeddings in the multimodal KB:

```bash
MODEL=google/siglip2-giant-opt-patch16-384 PORT=8013 scripts/run_image_embed_server.sh
MODEL=Qwen/Qwen3-Embedding-0.6B PORT=8012 scripts/run_text_embed_server.sh
```

Set the matching API endpoints in `.env`:

```bash
IMAGE_EMBED_API_BASE=http://<host>:8013
QWEN_TEXT_EMBED_API_BASE=http://<host>:8012
PMSR_KB=outputs/indexes/wikipedia_pmsr.index
PMSR_METADATA=outputs/indexes/wikipedia_pmsr_metadata.csv
```

6. Build the PMSR FAISS index:

```bash
set -a
source .env
set +a

python scripts/create_knowledge_base.py \
  --input-jsonl Wiki6M_ver_1_0_updated.jsonl \
  --index-output outputs/indexes/wikipedia_pmsr.index \
  --metadata-output outputs/indexes/wikipedia_pmsr_metadata.csv \
  --image-field image_path \
  --text-field wikipedia_summary \
  --caption-field wikipedia_summary \
  --fusion concat \
  --batch-size 32
```

The metadata CSV stores `image_path`, `caption`, and `text` fields. For Wikipedia, we use `wikipedia_summary` for both the text embedding input and the saved caption so retrieved image-text pairs can be passed directly to the PMSR agent.

### Text KB

For text-only retrieval, start an E5-compatible text embedding server and build or point to a text FAISS index:

```bash
MODEL=intfloat/e5-base-v2 PORT=8011 scripts/run_text_embed_server.sh
```

Set the text KB paths in `.env`:

```bash
TEXT_EMBED_API_BASE=http://<host>:8011
TEXT_KB=/path/to/wiki_text.index
TEXT_METADATA=/path/to/wiki_text.jsonl
```

## Citation

If you find PMSR useful for your research, please cite our paper:

```bibtex
@article{choi2025progressive,
  title   = {Progressive Multimodal Search and Reasoning for Knowledge-Intensive Visual Question Answering},
  author  = {Choi, Changin and Lee, Wonseok and Ko, Jungmin and Rhee, Wonjong},
  journal = {arXiv preprint arXiv:2509.00798},
  year    = {2025}
}
```

## Acknowledgements

We thank the authors and maintainers of the datasets, benchmarks, retrieval systems, and multimodal models used in this work, including Encyclopedic-VQA, InfoSeek, OK-VQA, FVQA, MMSearch, LiveVQA, Qwen-VL, SigLIP2, E5, FlashRAG, and related multimodal RAG and search-agent repositories.
