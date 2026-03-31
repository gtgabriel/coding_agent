"""Ollama backend — thin wrapper around native /api/chat with tool support."""

import json
import aiohttp
from dataclasses import dataclass


@dataclass
class ToolUse:
    id: str
    name: str
    input: dict


@dataclass
class TextBlock:
    text: str


@dataclass
class LLMResponse:
    """Normalized response from Ollama."""
    stop_reason: str   # "end_turn" or "tool_use"
    content: list      # list of TextBlock and/or ToolUse


class OllamaClient:
    """Async Ollama client using native /api/chat."""

    def __init__(
        self,
        host: str = "http://127.0.0.1:11434",
        model: str = "qwen3.5:35b-a3b",
        num_ctx: int = 8192,
        max_tokens: int = 4096,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens

    def _to_ollama_tools(self, tools: list) -> list:
        """Convert Anthropic-format tool defs to Ollama format."""
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
        return result

    def _to_ollama_messages(self, system: str, messages: list) -> list:
        """Convert internal message format to Ollama format."""
        result = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            # Tool results: list of {"type": "tool_result", ...}
            if role == "user" and isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        result.append({
                            "role": "tool",
                            "content": item.get("content", ""),
                            "tool_call_id": item.get("tool_use_id", ""),
                        })
                continue

            # Assistant with tool calls in content
            if role == "assistant" and isinstance(content, list):
                text_parts = []
                tool_calls = []
                for item in content:
                    if isinstance(item, TextBlock):
                        text_parts.append(item.text)
                    elif isinstance(item, ToolUse):
                        tool_calls.append({
                            "function": {
                                "name": item.name,
                                "arguments": item.input,
                            },
                        })
                msg_out = {"role": "assistant", "content": "\n".join(text_parts) or ""}
                if tool_calls:
                    msg_out["tool_calls"] = tool_calls
                result.append(msg_out)
                continue

            # Plain text message
            result.append({
                "role": role,
                "content": content if isinstance(content, str) else str(content),
            })

        return result

    async def chat(self, system: str, messages: list, tools: list) -> LLMResponse:
        """Send a chat request to Ollama and return a normalized response."""
        ollama_messages = self._to_ollama_messages(system, messages)
        ollama_tools = self._to_ollama_tools(tools) if tools else []

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                data = await resp.json()

        if "error" in data:
            return LLMResponse(
                stop_reason="end_turn",
                content=[TextBlock(text=f"Ollama error: {data['error']}")],
            )

        msg = data.get("message", {})
        content = []

        if msg.get("content"):
            # Strip thinking tags if present
            text = msg["content"]
            # Remove <think>...</think> blocks from output
            import re
            text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
            if text:
                content.append(TextBlock(text=text))

        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                content.append(ToolUse(
                    id=f"call_{fn['name']}",
                    name=fn["name"],
                    input=args,
                ))

        stop_reason = "tool_use" if msg.get("tool_calls") else "end_turn"
        return LLMResponse(stop_reason=stop_reason, content=content)
