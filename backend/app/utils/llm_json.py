# -*- coding: utf-8 -*-
"""LLM 调用的超时、重试与 JSON 解析修复工具。"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from app.core.log import logger


def strip_code_fence(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` ... ``` 包裹。"""
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json(text: str) -> Any:
    """从模型输出中尽量提取一个 JSON 对象/数组。

    处理顺序：
    1. 直接 json.loads；
    2. 去掉 Markdown 代码块；
    3. 去掉尾随逗号；
    4. 从第一个 { 或 [ 开始 raw_decode；
    5. 逐个尝试文本中出现的 { / [。
    """
    if not text or not isinstance(text, str):
        raise ValueError("LLM 输出为空，无法解析 JSON")

    cleaned = strip_code_fence(text)

    candidates = [cleaned]
    # 去掉 JSON 中常见的尾随逗号，例如 [1, 2,]
    candidates.append(re.sub(r",\s*([}\]])", r"\1", cleaned))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except Exception:
            continue

    raise ValueError(f"无法从模型输出中解析 JSON：{cleaned[:200]}")


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        content = "".join(parts)
    if content is None:
        return ""
    return str(content).strip()


async def ainvoke_text(
    llm: Any,
    prompt: str,
    *,
    timeout: float = 30.0,
    retries: int = 1,
    node_name: str = "llm",
) -> str:
    """带超时与重试的文本调用。"""
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout)
            text = _response_text(response)
            if not text:
                raise ValueError("LLM 返回空内容")
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                f"[{node_name}] LLM 调用失败（第 {attempt + 1}/{retries + 1} 次）："
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < retries:
                await asyncio.sleep(0.5)

    assert last_error is not None
    raise last_error


async def ainvoke_json(
    llm: Any,
    prompt: str,
    *,
    timeout: float = 30.0,
    retries: int = 1,
    node_name: str = "llm",
) -> Any:
    """带超时、重试与 JSON 修复的调用。"""
    last_error: Exception | None = None
    current_prompt = prompt

    for attempt in range(retries + 1):
        try:
            text = await ainvoke_text(
                llm,
                current_prompt,
                timeout=timeout,
                retries=0,
                node_name=f"{node_name}:json",
            )
            return extract_json(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                f"[{node_name}] JSON 解析失败（第 {attempt + 1}/{retries + 1} 次）："
                f"{type(exc).__name__}: {exc}"
            )
            if attempt < retries:
                current_prompt = (
                    prompt
                    + "\n\n上一次输出不是合法 JSON。请只输出合法 JSON，"
                    "不要解释、不要 Markdown 代码块、不要任何前后缀。"
                )
                await asyncio.sleep(0.5)

    assert last_error is not None
    raise last_error
