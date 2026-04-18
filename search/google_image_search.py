from __future__ import annotations

import base64
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import ollama
import requests
from ollama import Client

from agents.schemas import Evidence, SearchResult
from search.base_search import BaseSearch

try:
    from io import BytesIO

    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class GoogleImageSearch(BaseSearch):
    """Google Lens-style image search through ScrapingDog with optional Ollama summaries."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_url: str = "https://api.scrapingdog.com/google_lens",
        upload_bucket: str | None = None,
        upload_region: str | None = None,
        ollama_api_key: str | None = None,
        ollama_model: str = "gpt-oss:120b",
        summarize: bool = True,
        timeout: int = 60,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("SCRAPINGDOG_API_KEY", "")
        self.api_url = api_url
        self.upload_bucket = upload_bucket or os.environ.get("GOOGLE_LENS_UPLOAD_BUCKET", "images0707")
        self.upload_region = upload_region or os.environ.get("GOOGLE_LENS_UPLOAD_REGION", "ap-southeast-2")
        self.ollama_api_key = ollama_api_key if ollama_api_key is not None else os.environ.get("OLLAMA_API_KEY", "")
        self.ollama_model = ollama_model
        self.summarize = summarize
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.ollama_client: Client | None = None
        if self.ollama_api_key:
            os.environ.setdefault("OLLAMA_API_KEY", self.ollama_api_key)
            self.ollama_client = Client(
                host="https://ollama.com",
                headers={"Authorization": f"Bearer {self.ollama_api_key}"},
            )
        if not self.api_key:
            raise ValueError("Missing ScrapingDog API key. Pass api_key or set SCRAPINGDOG_API_KEY.")

    def search(self, query: Any, top_k: int = 5) -> list[SearchResult]:
        computed_query = self.compute_query(query)
        payload = self.search_with_google_lens(computed_query["lens_url"])
        return self.parse_results(
            payload,
            question=computed_query.get("question", ""),
            top_k=top_k,
            query=computed_query["image_url"],
        )

    def compute_query(self, query: Any) -> dict[str, str]:
        if isinstance(query, dict):
            value = query.get("image_url") or query.get("url") or query.get("image_path")
            question = str(query.get("question") or query.get("caption") or "")
        else:
            value = query
            question = ""
        image_url = str(value or "").strip()
        if not image_url:
            raise ValueError("Google image search requires an image path or URL.")
        if not image_url.startswith(("http://", "https://")):
            image_url = self.upload_image_path(image_url)
        return {
            "image_url": image_url,
            "lens_url": f"https://lens.google.com/uploadbyurl?url={image_url}",
            "question": question,
        }

    def parse_results(
        self,
        response_data: dict[str, Any],
        *,
        question: str = "",
        top_k: int = 10,
        query: str = "",
    ) -> list[SearchResult]:
        raw_results = response_data.get("lens_results") or response_data.get("visual_matches") or []
        results: list[SearchResult] = []
        for item in raw_results:
            if len(results) >= top_k:
                break
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "")
            url = str(item.get("link") or item.get("url") or "")
            source_name = str(item.get("source") or "")
            thumbnail = str(item.get("original_thumbnail") or item.get("thumbnail") or "")
            if thumbnail and self._is_invalid_thumbnail(thumbnail):
                continue
            content = ""
            summary = title
            if self.ollama_client is not None and url and self.summarize:
                content, summary = self._fetch_and_summarize(url, title, source_name, question)
            rank = len(results) + 1
            evidence = Evidence(
                source="google_image",
                modality="image",
                title=title,
                text=summary,
                url=url,
                image_path=thumbnail,
                caption=title,
                score=1.0 / rank,
                rank=rank,
                metadata={
                    "source_name": source_name,
                    "uploaded_image_url": query,
                    "question": question,
                    "content": content,
                    "raw": item,
                },
            )
            results.append(SearchResult(evidence=evidence, query=query, search_type="google_image"))
        return results

    def upload_image_path(self, image_path: str | Path) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return self.upload_base64_image(encoded, original_name=path.name)

    def upload_base64_image(self, base64_image: str, *, original_name: str = "query.jpg") -> str:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("Local Google Lens image search requires boto3 for S3 upload.") from exc

        image_data = base64.b64decode(base64_image.split(",")[-1])
        suffix = Path(original_name).suffix.lower() or ".jpg"
        digest = hashlib.sha256(image_data).hexdigest()[:24]
        object_key = f"{digest}{suffix}"
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=self.upload_bucket,
            Key=object_key,
            Body=image_data,
            ContentType="image/jpeg",
        )
        return f"https://{self.upload_bucket}.s3.{self.upload_region}.amazonaws.com/{object_key}"

    def search_with_google_lens(self, lens_url: str) -> dict[str, Any]:
        request_url = f"{self.api_url}?api_key={self.api_key}&url={lens_url}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.get(request_url, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise last_error or RuntimeError("Google Lens search failed.")

    def _fetch_and_summarize(self, url: str, title: str, source_name: str, question: str) -> tuple[str, str]:
        try:
            web_response = ollama.web_search(url)
            raw_results = getattr(web_response, "results", []) or []
            if not raw_results:
                return "", title
            content = str(getattr(raw_results[0], "content", "") or "")
            if not content:
                return "", title
            prompt = (
                f"Given the search query: '{question}'\n\n"
                "Summarize the following web content concisely in a single paragraph.\n\n"
                f"Title: {title}\nSource: {source_name}\n\nContent: {content[:120000]}"
            )
            response = self.ollama_client.chat(
                self.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return content, response["message"]["content"]
        except Exception:
            return "", title

    def _is_invalid_thumbnail(self, image_url: str) -> bool:
        if not PIL_AVAILABLE:
            return False
        try:
            response = requests.get(image_url, timeout=5)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            return img.size == (1, 1)
        except Exception:
            return True
