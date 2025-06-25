import os
import requests
from ..types import MessageList, SamplerBase, SamplerResponse

class DeepSeekSampler(SamplerBase):
    """Sampler for DeepSeek models via a chat completion style API."""

    def __init__(self, *, model: str = "deepseek-chat", api_key: str | None = None, base_url: str = "https://api.deepseek.com/v1/chat/completions", temperature: float = 0.5, max_tokens: int = 1024):
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if self.api_key is None:
            raise ValueError("DEEPSEEK_API_KEY must be set")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        data = {
            "model": self.model,
            "messages": message_list,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(self.base_url, json=data, headers=headers)
        resp.raise_for_status()
        result = resp.json()
        try:
            text = result["choices"][0]["message"]["content"]
        except Exception:
            raise ValueError(f"Invalid response from DeepSeek API: {result}")
        return SamplerResponse(response_text=text, response_metadata={"api_response": result}, actual_queried_message_list=message_list)
