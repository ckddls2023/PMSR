# Knowledge Base Preparation

PMSR retrieves from textual and multimodal knowledge bases. For the multimodal Wikipedia KB, we pair Wikipedia text entries with local Wikipedia entity images, encode the image and text fields with embedding API servers, concatenate the embeddings, and write a FAISS index plus metadata.

## Wikipedia Image-Text KB

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

## Text KB

For text-only retrieval, use the Wikipedia corpus from the [FlashRAG retrieval corpus](https://www.modelscope.cn/datasets/hhjinjiajie/FlashRAG_Dataset/tree/master/retrieval_corpus). Our experiments use `wiki18_100w.jsonl` encoded with `intfloat/e5-base-v2`.

Start an E5-compatible text embedding server:

```bash
MODEL=intfloat/e5-base-v2 PORT=8011 scripts/run_text_embed_server.sh
```

Build or point to the matching E5 FAISS index and metadata file. In `.env`, set:

```bash
TEXT_EMBED_API_BASE=http://<host>:8011
TEXT_KB=/path/to/wiki18_100w_e5_flat.index
TEXT_METADATA=/path/to/wiki18_100w.jsonl
```
