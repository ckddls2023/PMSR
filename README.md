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
+-- data.md
+-- KB.md
`-- README.md
```

## Data and Knowledge Bases

PMSR expects local benchmark data and FAISS knowledge bases prepared before evaluation.

- See [data.md](data.md) for dataset preparation instructions for InfoSeek, E-VQA, LiveVQA, FVQA, InfoSeek Human, and MMSearch.
- See [KB.md](KB.md) for Wikipedia image-text and text-only knowledge-base preparation.

The evaluation entry point reads any prepared PMSR JSONL file directly:

```bash
python eval/main.py \
  --data data/EVQA_test.jsonl \
  --model Qwen/Qwen3.5-9B \
  --api_base http://<host>:8004/ \
  --itercount 3 \
  --topk 10
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
