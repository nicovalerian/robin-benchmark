import json
from dataclasses import dataclass
from typing import Any

import aiohttp


JUDGE_PROMPT_TEMPLATE = """Anda adalah evaluator yang menilai kualitas respons AI terhadap instruksi.

## Instruksi Asli:
{instruction}

## Respons yang Dievaluasi:
{response}

## Referensi Jawaban (Gold Standard):
{reference}

## Kriteria Penilaian:
1. **Pemahaman Instruksi**: Apakah respons memahami maksud instruksi?
2. **Kelengkapan**: Apakah semua bagian instruksi dijawab?
3. **Kepatuhan Constraint**: Apakah batasan (kata kunci, panjang) dipenuhi?
4. **Kualitas Konten**: Apakah informasi akurat dan relevan?
5. **Kesesuaian Format**: Apakah format sesuai permintaan?

## Instruksi untuk Evaluator:
Berikan skor 1-5 dengan kriteria:
- 1: Gagal total - tidak memahami instruksi sama sekali
- 2: Buruk - memahami sebagian tapi banyak kesalahan
- 3: Cukup - memenuhi sebagian besar instruksi dengan beberapa kekurangan
- 4: Baik - memenuhi instruksi dengan baik, sedikit kekurangan minor
- 5: Sangat Baik - memenuhi semua instruksi dengan sempurna

Berikan respons dalam format JSON:
{{"score": <1-5>, "reasoning": "<penjelasan singkat>"}}
"""


@dataclass
class JudgeResult:
    score: int
    reasoning: str
    raw_response: str
    success: bool
    error_message: str | None = None


class LLMJudge:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        provider: str = "openai",
    ):
        self.api_key = api_key
        self.model = model
        self.provider = provider
    
    async def judge(
        self,
        instruction: str,
        response: str,
        reference: str = "",
    ) -> JudgeResult:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            instruction=instruction,
            response=response,
            reference=reference or "(Tidak ada referensi)",
        )
        
        if self.provider == "openai":
            return await self._judge_openai(prompt)
        elif self.provider == "anthropic":
            return await self._judge_anthropic(prompt)
        else:
            return JudgeResult(
                score=0,
                reasoning="",
                raw_response="",
                success=False,
                error_message=f"Unsupported provider: {self.provider}",
            )
    
    async def _judge_openai(self, prompt: str) -> JudgeResult:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 500,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        return JudgeResult(
                            score=0,
                            reasoning="",
                            raw_response=str(data),
                            success=False,
                            error_message=data.get("error", {}).get("message", "API error"),
                        )
                    
                    raw = data["choices"][0]["message"]["content"]
                    return self._parse_response(raw)
        except Exception as e:
            return JudgeResult(
                score=0,
                reasoning="",
                raw_response="",
                success=False,
                error_message=str(e),
            )
    
    async def _judge_anthropic(self, prompt: str) -> JudgeResult:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    data = await resp.json()
                    
                    if resp.status != 200:
                        return JudgeResult(
                            score=0,
                            reasoning="",
                            raw_response=str(data),
                            success=False,
                            error_message=str(data),
                        )
                    
                    raw = data["content"][0]["text"]
                    return self._parse_response(raw)
        except Exception as e:
            return JudgeResult(
                score=0,
                reasoning="",
                raw_response="",
                success=False,
                error_message=str(e),
            )
    
    def _parse_response(self, raw: str) -> JudgeResult:
        try:
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start != -1 and json_end > json_start:
                json_str = raw[json_start:json_end]
                parsed = json.loads(json_str)
                
                return JudgeResult(
                    score=int(parsed.get("score", 0)),
                    reasoning=parsed.get("reasoning", ""),
                    raw_response=raw,
                    success=True,
                )
        except Exception:
            pass
        
        return JudgeResult(
            score=0,
            reasoning="",
            raw_response=raw,
            success=False,
            error_message="Failed to parse judge response",
        )
    
    async def judge_batch(
        self,
        items: list[dict],
    ) -> list[JudgeResult]:
        import asyncio
        
        tasks = [
            self.judge(
                instruction=item["instruction"],
                response=item["response"],
                reference=item.get("reference", ""),
            )
            for item in items
        ]
        
        return await asyncio.gather(*tasks)
