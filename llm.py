"""Ollama backend — thin wrapper around native /api/chat with tool support."""

import asyncio
import json
import aiohttp
from dataclasses import dataclass
from urllib.request import urlopen, Request


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
    prompt_tokens: int = 0   # tokens used by the prompt
    output_tokens: int = 0   # tokens generated in response


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
                            "id": item.id,
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

    async def chat(self, system: str, messages: list, tools: list,
                   on_thinking: callable = None, on_content: callable = None,
                   cancel_event=None) -> LLMResponse:
        """Send a streaming chat request to Ollama and return a normalized response.

        Callbacks (called from a background thread):
            on_thinking(snippet: str) — called periodically with thinking text
            on_content(token: str)    — called for each content token
        """
        ollama_messages = self._to_ollama_messages(system, messages)
        ollama_tools = self._to_ollama_tools(tools) if tools else []

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        if ollama_tools:
            payload["tools"] = ollama_tools

        def _do_stream(p):
            body = json.dumps(p).encode()
            req = Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
            resp = urlopen(req, timeout=300)

            thinking_buf = ""
            content_buf = ""
            tool_calls = []
            prompt_tokens = 0
            output_tokens = 0

            for raw_line in resp:
                if cancel_event and cancel_event.is_set():
                    resp.close()
                    return {"cancelled": True}

                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)

                if "error" in chunk:
                    return {"error": chunk["error"], "tools_sent": bool(ollama_tools)}

                msg = chunk.get("message", {})

                # Thinking tokens
                if msg.get("thinking"):
                    thinking_buf += msg["thinking"]
                    if on_thinking and len(thinking_buf) % 40 < len(msg["thinking"]):
                        # Extract last meaningful phrase
                        snippet = thinking_buf.rstrip().rsplit("\n", 1)[-1].strip()
                        if len(snippet) > 80:
                            snippet = snippet[-80:]
                        if snippet:
                            on_thinking(snippet)

                # Content tokens
                if msg.get("content"):
                    content_buf += msg["content"]
                    if on_content:
                        on_content(msg["content"])

                # Tool calls (arrive in final chunk)
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]

                # Final chunk
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count", 0)
                    output_tokens = chunk.get("eval_count", 0)
                    break

            resp.close()
            return {
                "content": content_buf,
                "thinking": thinking_buf,
                "tool_calls": tool_calls,
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
            }

        result = await asyncio.to_thread(_do_stream, payload)

        # Handle cancellation
        if result.get("cancelled"):
            return LLMResponse(stop_reason="end_turn", content=[])

        # Handle errors
        if "error" in result:
            err = result["error"]
            if "does not support tools" in err and result.get("tools_sent"):
                payload.pop("tools", None)
                payload["stream"] = False
                def _fallback(p):
                    body = json.dumps(p).encode()
                    req = Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
                    with urlopen(req, timeout=300) as resp:
                        return json.loads(resp.read())
                data = await asyncio.to_thread(_fallback, payload)
                if "error" not in data:
                    text = (data.get("message", {}).get("content") or "").strip()
                    return LLMResponse(
                        stop_reason="end_turn",
                        content=[TextBlock(text=text)] if text else [],
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        output_tokens=data.get("eval_count", 0),
                    )
                return LLMResponse(stop_reason="end_turn", content=[TextBlock(text=f"Ollama error: {data['error']}")])
            return LLMResponse(stop_reason="end_turn", content=[TextBlock(text=f"Ollama error: {err}")])

        # Build response
        content = []
        import re

        text = result["content"]
        if not result["thinking"]:
            # Strip embedded thinking tags if Ollama didn't separate them
            text = re.sub(r"<think>[\s\S]*?</think>", "", text)
            text = re.sub(r"<\|channel>[\s\S]*?<channel\|>", "", text)
            text = re.sub(r"<\|channel>[\s\S]*", "", text)
            text = re.sub(r"<channel\|>", "", text)

        text = text.strip()
        if text:
            content.append(TextBlock(text=text))

        if result["tool_calls"]:
            for tc in result["tool_calls"]:
                fn = tc["function"]
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                content.append(ToolUse(
                    id=tc.get("id") or f"call_{fn['name']}",
                    name=fn["name"],
                    input=args,
                ))

        stop_reason = "tool_use" if result["tool_calls"] else "end_turn"
        return LLMResponse(
            stop_reason=stop_reason,
            content=content,
            prompt_tokens=result["prompt_tokens"],
            output_tokens=result["output_tokens"],
        )
