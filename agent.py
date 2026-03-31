#!/usr/bin/env python3
"""Qwen Coding Agent — a terminal-based coding assistant powered by local Ollama."""

import asyncio
import os
import signal
import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style  # styling the prompt

from llm import OllamaClient, TextBlock, ToolUse, LLMResponse
from tools import TOOL_DEFINITIONS, execute_tool
from prompts import SYSTEM_PROMPT

console = Console()

# ── Config ─────────────────────────────────────────────────────────

MODEL = os.environ.get("QWEN_MODEL", "qwen3.5:35b-a3b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
NUM_CTX = int(os.environ.get("QWEN_NUM_CTX", "8192"))
MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "4096"))
MAX_TURNS = int(os.environ.get("QWEN_MAX_TURNS", "20"))
COMPACT_THRESHOLD = int(os.environ.get("QWEN_COMPACT_THRESHOLD", "20"))
COMPACT_KEEP = int(os.environ.get("QWEN_COMPACT_KEEP", "6"))

# ── Cancellation ──────────────────────────────────────────────────

# The currently running agent task — set during agent_loop, None at prompt
_active_task: asyncio.Task | None = None


def _install_signal_handlers(loop: asyncio.AbstractEventLoop):
    """Install signal handlers for SIGINT (cancel) and SIGWINCH (resize)."""
    def sigint_handler():
        if _active_task and not _active_task.done():
            _active_task.cancel()
    loop.add_signal_handler(signal.SIGINT, sigint_handler)

    def sigwinch_handler():
        # Clear the screen and reposition cursor on resize
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
        console.print("[dim]Terminal resized. Output cleared.[/]\n")
    loop.add_signal_handler(signal.SIGWINCH, sigwinch_handler)


# ── Compaction ─────────────────────────────────────────────────────


async def compact_history(client: OllamaClient, messages: list) -> list:
    """Summarize older messages, keep recent ones intact. Returns new message list."""
    if len(messages) <= COMPACT_THRESHOLD:
        return messages

    old = messages[:-COMPACT_KEEP]
    recent = messages[-COMPACT_KEEP:]

    old_text = []
    for msg in old:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            old_text.append(f"[{role}]: {content[:300]}")
        elif isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    parts.append(f"tool_result: {str(item.get('content', ''))[:100]}")
                elif isinstance(item, TextBlock):
                    parts.append(item.text[:100])
                elif isinstance(item, ToolUse):
                    parts.append(f"tool_call: {item.name}({str(item.input)[:80]})")
            old_text.append(f"[{role}]: {'; '.join(parts)}")

    summary_prompt = (
        "Summarize this conversation history in 2-3 concise paragraphs. "
        "Focus on: what the user asked for, what files were created/modified, "
        "what commands were run, and the current state of the work.\n\n"
        + "\n".join(old_text)
    )

    with console.status("[bold yellow]Compacting conversation history...[/]", spinner="dots"):
        resp = await client.chat(
            system="You are a conversation summarizer. Be concise and factual.",
            messages=[{"role": "user", "content": summary_prompt}],
            tools=[],
        )

    summary = "\n".join(b.text for b in resp.content if isinstance(b, TextBlock))
    if not summary:
        summary = "Previous conversation context was too large to summarize."

    compacted = [
        {"role": "user", "content": f"[Previous conversation summary]: {summary}"},
        {"role": "assistant", "content": "Understood. I have context from our previous conversation. How can I help?"},
    ] + recent

    console.print(f"  [dim]Compacted {len(old)} old messages into summary, kept {len(recent)} recent[/]")
    return compacted


# ── Tool confirmation ──────────────────────────────────────────────

# Bash commands that need confirmation
_DANGEROUS_PREFIXES = (
    "rm ", "rm\t", "rmdir ", "mv ", "cp ",
    "git push", "git reset", "git checkout .",
    "git clean", "git branch -D", "git branch -d",
    "chmod ", "chown ", "sudo ", "kill ", "pkill ",
    "dd ", "> ", ">> ",
)


def _needs_confirmation(tool_name: str, args: dict) -> str | None:
    """Return a description if this tool call needs user confirmation, else None."""
    if tool_name == "write_file":
        return f"Write to [cyan]{args.get('path', '?')}[/]"
    if tool_name == "edit_file":
        return f"Edit [cyan]{args.get('path', '?')}[/]"
    if tool_name == "bash":
        cmd = args.get("command", "")
        for prefix in _DANGEROUS_PREFIXES:
            if cmd.strip().startswith(prefix) or f"&& {prefix}" in cmd or f"; {prefix}" in cmd:
                return f"Run: [yellow]{cmd}[/]"
    return None


async def _confirm(description: str) -> bool:
    """Ask the user to confirm a tool action. Returns True if approved."""
    console.print(f"  [bold yellow]?[/] {description}")
    try:
        answer = await asyncio.to_thread(
            input, "    Allow? [y/N] "
        )
        return answer.strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ── Agent loop ─────────────────────────────────────────────────────


async def agent_loop(client: OllamaClient, messages: list, user_input: str) -> str:
    """Run the agentic tool-use loop for a single user message. Returns final text."""
    messages.append({"role": "user", "content": user_input})

    for turn in range(MAX_TURNS):
        with console.status(
            f"[bold blue]Thinking (turn {turn + 1})... [dim]Ctrl+C to cancel[/][/]",
            spinner="dots",
        ):
            response = await client.chat(
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )

        if response.stop_reason == "end_turn":
            text = "\n".join(b.text for b in response.content if isinstance(b, TextBlock))
            messages.append({"role": "assistant", "content": text})
            return text

        tool_results = []
        for block in response.content:
            if isinstance(block, TextBlock) and block.text:
                console.print(Markdown(block.text))

            if isinstance(block, ToolUse):
                args_display = _format_tool_args(block.name, block.input)
                console.print(
                    f"  [dim]>[/] [bold purple]{block.name}[/] {args_display}"
                )

                # Ask for confirmation on writes/edits/dangerous commands
                confirm_desc = _needs_confirmation(block.name, block.input)
                if confirm_desc and not await _confirm(confirm_desc):
                    result = "User denied this action."
                else:
                    result = await execute_tool(block.name, block.input)

                preview = result[:200] + "..." if len(result) > 200 else result
                for line in preview.splitlines()[:5]:
                    console.print(Text(f"    {line}", style="dim"))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

        if not tool_results:
            text = "\n".join(b.text for b in response.content if isinstance(b, TextBlock))
            messages.append({"role": "assistant", "content": text})
            return text

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Error: max turns reached."


def _cleanup_messages(messages: list, user_input: str):
    """Remove incomplete messages from a cancelled agent run."""
    # Walk back and remove everything added during this run
    while messages and messages[-1].get("content") != user_input:
        messages.pop()
    # Remove the user message itself
    if messages and messages[-1].get("content") == user_input:
        messages.pop()


def _format_tool_args(name: str, args: dict) -> str:
    """Format tool arguments for display."""
    if name == "bash":
        return f"[cyan]{args.get('command', '')}[/]"
    if name == "read_file":
        extra = ""
        if args.get("offset"):
            extra += f" (from line {args['offset']})"
        if args.get("limit"):
            extra += f" ({args['limit']} lines)"
        return f"[cyan]{args.get('path', '')}{extra}[/]"
    if name == "write_file":
        return f"[cyan]{args.get('path', '')}[/]"
    if name == "edit_file":
        return f"[cyan]{args.get('path', '')}[/]"
    if name == "list_files":
        return f"[cyan]{args.get('pattern', '')}[/]"
    if name == "web_search":
        return f"[cyan]{args.get('query', '')}[/]"
    if name == "browse_url":
        return f"[cyan]{args.get('url', '')}[/]"
    if name == "analyze_image":
        return f"[cyan]{args.get('path', '')}[/] [dim]({args.get('question', 'describe')})[/]"
    return str(args)[:80]


# ── REPL ───────────────────────────────────────────────────────────


async def main():
    global _active_task

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)

    console.print(
        Panel(
            f"[bold]Qwen Coding Agent[/]\n"
            f"Model: [cyan]{MODEL}[/]  Ctx: [cyan]{NUM_CTX}[/]\n"
            f"[dim]exit · clear · Ctrl+C to cancel[/]",
            border_style="blue",
            width=40,
        )
    )

    client = OllamaClient(
        host=OLLAMA_HOST,
        model=MODEL,
        num_ctx=NUM_CTX,
        max_tokens=MAX_TOKENS,
    )

    messages = []

    history_path = os.path.expanduser("~/.qwen-agent-history")
    
    # Create a style that makes everything yellow
    yellow_style = Style.from_dict({
        '': 'ansiyellow', 
    })

    # Now update this line to include the style
    session = PromptSession(history=FileHistory(history_path), style=yellow_style)

    # session = PromptSession(history=FileHistory(history_path))

    while True:
        try:
            user_input = await asyncio.to_thread(session.prompt, "Human: ", multiline=False)
        except EOFError:
            console.print("\n[dim]Bye![/]")
            break
        except KeyboardInterrupt:
            continue  # Ctrl+C at prompt just resets the input line

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Bye![/]")
            break
        if user_input.lower() == "clear":
            messages.clear()
            console.print("[dim]Conversation cleared.[/]")
            continue

        # Run agent — Ctrl+C here cancels the task, not the app
        messages[:] = await compact_history(client, messages)

        _active_task = asyncio.current_task()
        try:
            result = await agent_loop(client, messages, user_input)
            if result:
                console.print()
                console.print(Markdown(result))
        except asyncio.CancelledError:
            _cleanup_messages(messages, user_input)
            console.print("\n[yellow]Cancelled.[/]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")
        finally:
            _active_task = None


if __name__ == "__main__":
    asyncio.run(main())
