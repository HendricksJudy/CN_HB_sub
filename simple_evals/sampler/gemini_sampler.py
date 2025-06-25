import os
import requests
from ..types import MessageList, SamplerBase, SamplerResponse

class GeminiChatSampler(SamplerBase):
    """Sampler for Google's Gemini API."""

    def __init__(self, *, model: str = "models/gemini-1.5-pro-latest", api_key: str | None = None, temperature: float = 0.5, max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self.api_key is None:
            raise ValueError("GEMINI_API_KEY must be set")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.url = f"https://generativelanguage.googleapis.com/v1beta/{self.model}:generateContent"

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        contents = []
        for m in message_list:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})
        data = {
            "contents": contents,
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_tokens},
        }
        resp = requests.post(self.url, params={"key": self.api_key}, json=data)
        resp.raise_for_status()
        result = resp.json()
        try:
            text = result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            raise ValueError(f"Invalid response from Gemini API: {result}")
        return SamplerResponse(response_text=text, response_metadata={"api_response": result}, actual_queried_message_list=message_list)
