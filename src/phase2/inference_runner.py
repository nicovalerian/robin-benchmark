import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

import aiohttp


@dataclass
class InferenceResult:
    model_name: str
    prompt: str
    response: str
    tokens_used: int
    latency_ms: float
    success: bool
    error_message: str | None = None


class BaseProvider(ABC):
    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        pass


class OpenAIProvider(BaseProvider):
    BASE_URL = "https://api.openai.com/v1/chat/completions"
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        return InferenceResult(
                            model_name=self.model_id,
                            prompt=prompt,
                            response="",
                            tokens_used=0,
                            latency_ms=(time.perf_counter() - start_time) * 1000,
                            success=False,
                            error_message=data.get("error", {}).get("message", "Unknown error"),
                        )
                    
                    response_text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=response_text,
                        tokens_used=tokens,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        success=True,
                    )
        except Exception as e:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class AnthropicProvider(BaseProvider):
    BASE_URL = "https://api.anthropic.com/v1/messages"
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        return InferenceResult(
                            model_name=self.model_id,
                            prompt=prompt,
                            response="",
                            tokens_used=0,
                            latency_ms=(time.perf_counter() - start_time) * 1000,
                            success=False,
                            error_message=data.get("error", {}).get("message", "Unknown error"),
                        )
                    
                    response_text = data["content"][0]["text"]
                    input_tokens = data.get("usage", {}).get("input_tokens", 0)
                    output_tokens = data.get("usage", {}).get("output_tokens", 0)
                    
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=response_text,
                        tokens_used=input_tokens + output_tokens,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        success=True,
                    )
        except Exception as e:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class TogetherProvider(BaseProvider):
    BASE_URL = "https://api.together.xyz/v1/chat/completions"
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        return InferenceResult(
                            model_name=self.model_id,
                            prompt=prompt,
                            response="",
                            tokens_used=0,
                            latency_ms=(time.perf_counter() - start_time) * 1000,
                            success=False,
                            error_message=str(data),
                        )
                    
                    response_text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=response_text,
                        tokens_used=tokens,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        success=True,
                    )
        except Exception as e:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class GoogleProvider(BaseProvider):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        
        url = f"{self.BASE_URL}/{self.model_id}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        return InferenceResult(
                            model_name=self.model_id,
                            prompt=prompt,
                            response="",
                            tokens_used=0,
                            latency_ms=(time.perf_counter() - start_time) * 1000,
                            success=False,
                            error_message=str(data),
                        )
                    
                    response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=response_text,
                        tokens_used=0,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        success=True,
                    )
        except Exception as e:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class GroqProvider(BaseProvider):
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        return InferenceResult(
                            model_name=self.model_id,
                            prompt=prompt,
                            response="",
                            tokens_used=0,
                            latency_ms=(time.perf_counter() - start_time) * 1000,
                            success=False,
                            error_message=data.get("error", {}).get("message", str(data)),
                        )
                    
                    response_text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=response_text,
                        tokens_used=tokens,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        success=True,
                    )
        except Exception as e:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class OpenRouterProvider(BaseProvider):
    """OpenRouter - unified API for GPT-4, Claude, Gemini, Llama, Qwen, etc."""
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nicovalerian/robin-benchmark",
            "X-Title": "ROBIN Benchmark",
        }
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        error_msg = data.get("error", {})
                        if isinstance(error_msg, dict):
                            error_msg = error_msg.get("message", str(data))
                        return InferenceResult(
                            model_name=self.model_id,
                            prompt=prompt,
                            response="",
                            tokens_used=0,
                            latency_ms=(time.perf_counter() - start_time) * 1000,
                            success=False,
                            error_message=str(error_msg),
                        )
                    
                    response_text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=response_text,
                        tokens_used=tokens,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        success=True,
                    )
        except Exception as e:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


class LocalProvider(BaseProvider):
    def __init__(self, api_key: str, model_id: str, base_url: str = "http://localhost:11434"):
        super().__init__(api_key, model_id)
        self.base_url = base_url.rstrip("/")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> InferenceResult:
        start_time = time.perf_counter()
        
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key not in ("none", "local", ""):
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        url = f"{self.base_url}/v1/chat/completions"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120, connect=10),
                ) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        error_msg = data.get("error", {})
                        if isinstance(error_msg, dict):
                            error_msg = error_msg.get("message", str(data))
                        return InferenceResult(
                            model_name=self.model_id,
                            prompt=prompt,
                            response="",
                            tokens_used=0,
                            latency_ms=(time.perf_counter() - start_time) * 1000,
                            success=False,
                            error_message=str(error_msg),
                        )
                    
                    response_text = data["choices"][0]["message"]["content"]
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    
                    return InferenceResult(
                        model_name=self.model_id,
                        prompt=prompt,
                        response=response_text,
                        tokens_used=tokens,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        success=True,
                    )
        except aiohttp.ClientConnectorError:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=f"Cannot connect to {self.base_url}. Is the server running?",
            )
        except Exception as e:
            return InferenceResult(
                model_name=self.model_id,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )


PROVIDER_MAP = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "together": TogetherProvider,
    "google": GoogleProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "local": LocalProvider,
    "ollama": LocalProvider,
    "vllm": LocalProvider,
    "lmstudio": LocalProvider,
}


class InferenceRunner:
    def __init__(
        self,
        providers_config: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        max_concurrent: int = 5,
        rate_limit_delay: float = 0.5,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        
        self.providers: dict[str, BaseProvider] = {}
        self.rate_limit_delays: dict[str, float] = {}
        for config in providers_config:
            name = config["name"]
            provider_type = config["provider"]
            model_id = config["model_id"]
            api_key = config.get("api_key", "")
            base_url = config.get("base_url", "")
            
            provider_class = PROVIDER_MAP.get(provider_type)
            if provider_class:
                if provider_type in ("local", "ollama", "vllm", "lmstudio"):
                    default_urls = {
                        "ollama": "http://localhost:11434",
                        "vllm": "http://localhost:8000",
                        "lmstudio": "http://localhost:1234",
                        "local": "http://localhost:8000",
                    }
                    url = base_url or default_urls.get(provider_type, "http://localhost:8000")
                    self.providers[name] = provider_class(api_key, model_id, url)
                else:
                    self.providers[name] = provider_class(api_key, model_id)
            self.rate_limit_delays[name] = config.get("rate_limit_delay", self.rate_limit_delay)
    
    async def run_single(
        self,
        model_name: str,
        prompt: str,
    ) -> InferenceResult:
        if model_name not in self.providers:
            return InferenceResult(
                model_name=model_name,
                prompt=prompt,
                response="",
                tokens_used=0,
                latency_ms=0,
                success=False,
                error_message=f"Provider not configured: {model_name}",
            )
        
        provider = self.providers[model_name]
        result = await provider.generate(
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        delay = self.rate_limit_delays.get(model_name, self.rate_limit_delay)
        await asyncio.sleep(delay)
        return result
    
    async def run_batch(
        self,
        model_name: str,
        prompts: list[str],
    ) -> list[InferenceResult]:
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def limited_run(prompt: str) -> InferenceResult:
            async with semaphore:
                return await self.run_single(model_name, prompt)
        
        tasks = [limited_run(p) for p in prompts]
        return await asyncio.gather(*tasks)
    
    async def run_all_models(
        self,
        prompt: str,
    ) -> dict[str, InferenceResult]:
        results = {}
        for model_name in self.providers:
            results[model_name] = await self.run_single(model_name, prompt)
        return results
