"""Optional Responses HTTP transport; replace it to use another provider/API."""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


class LLMTransportError(RuntimeError):
    """Authentication, networking, refusal or unsupported API capability."""


def _http_error_detail(error: HTTPError, api_key: str) -> str:
    """Show structured errors or a bounded, credential-redacted body excerpt."""
    def clean(value, limit=600):
        value = value.replace(api_key, "[REDACTED]")
        value = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
        value = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", value)
        return " ".join(value.split())[:limit]

    try:
        body = error.read(16384)
    except OSError:
        return "error_body_unavailable"
    fields = [f"response_bytes={len(body)}"]
    if error.headers is not None:
        for header in ("content-type", "server", "x-request-id"):
            value = error.headers.get(header)
            if isinstance(value, str) and value:
                fields.append(f"{header}={clean(value, 160)}")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeError):
        payload = None
    details = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(details, dict):
        # Non-JSON gateways and proxies often return plain text or HTML.
        preview = body.decode("utf-8", errors="replace")
        preview = re.sub(r"<[^>]*>", " ", preview)
        if preview.strip():
            fields.append("body_preview=" + clean(preview, 400))
        else:
            fields.append("empty_error_body")
        return "; ".join(fields)
    for name in ("type", "code", "param", "message"):
        value = details.get(name)
        if not isinstance(value, str) or not value:
            continue
        fields.append(f"{name}={clean(value, 600 if name == 'message' else 160)}")
    return "; ".join(fields)


@dataclass
class ResponsesJSONProvider:
    model: str
    api_key: str = field(repr=False)
    endpoint: str = "https://api.openai.com/v1/responses"
    timeout: float = 90.0

    def __post_init__(self):
        if not self.model or not self.api_key:
            raise ValueError("model and API key are required")
        if not self.endpoint.startswith("https://"):
            raise ValueError("LLM endpoint must use HTTPS")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    @classmethod
    def from_env(cls, model: str | None = None):
        return cls(model=model or os.environ.get("SEMANTIC_LLM_MODEL", ""),
                   api_key=os.environ.get("SEMANTIC_LLM_API_KEY", ""),
                   endpoint=os.environ.get("SEMANTIC_LLM_ENDPOINT", "https://api.openai.com/v1/responses"))

    def complete(self, *, system_prompt, payload, schema, image_bytes=None, feedback=None):
        content = [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}]
        if image_bytes is not None:
            # Re-encode a valid RGB PNG rather than assuming input MIME type.
            with Image.open(BytesIO(image_bytes)) as image:
                buffer = BytesIO()
                image.convert("RGB").save(buffer, format="PNG")
            content.append({"type": "input_image", "image_url": "data:image/png;base64," +
                            base64.b64encode(buffer.getvalue()).decode("ascii")})
        if feedback:
            system_prompt += "\nThe previous response failed validation: " + feedback + "\nCorrect it using the original input."
        body = {"model": self.model, "store": False,
                "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
                "text": {"format": {"type": "json_schema", "name": "semantic_analysis", "strict": True, "schema": schema}}}
        request = Request(self.endpoint, data=json.dumps(body).encode("utf-8"),
                          headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as error:
            detail = _http_error_detail(error, self.api_key)
            message = f"LLM HTTP {error.code}"
            if detail:
                message += "; " + detail
            else:
                message += "; check model, credentials, endpoint and schema/vision support"
            raise LLMTransportError(message) from None
        except (URLError, TimeoutError) as error:
            raise LLMTransportError("LLM network request failed") from error
        except (ValueError, UnicodeError) as error:
            raise ValueError("malformed provider response JSON") from error
        if not isinstance(result, dict):
            raise ValueError("malformed provider response envelope")
        if result.get("status") != "completed":
            raise LLMTransportError("LLM response was incomplete or failed")
        try:
            parts = [part for item in result["output"] if item["type"] == "message" for part in item["content"]]
            if any(part["type"] == "refusal" for part in parts):
                raise LLMTransportError("LLM refused the analysis request")
            text = "".join(part["text"] for part in parts if part["type"] == "output_text")
        except (KeyError, TypeError) as error:
            raise ValueError("malformed provider response envelope") from error
        if not text:
            raise ValueError("provider returned no JSON text")
        return text
