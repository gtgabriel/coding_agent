#!/usr/bin/env python3
"""Vaib Kodar — a terminal-based coding assistant powered by local Ollama."""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from llm import OllamaClient, TextBlock, ToolUse, LLMResponse
from tools import TOOL_DEFINITIONS, execute_tool
from prompts import SYSTEM_PROMPT

# ── Session log ───────────────────────────────────────────────────

LOG_DIR = os.path.expanduser("~/.vaib-kodar/logs")
os.makedirs(LOG_DIR, exist_ok=True)
_log_path = os.path.join(LOG_DIR, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
_log_file = open(_log_path, "a")


def slog(event: str, **data):
    """Write a structured log entry."""
    entry = {"ts": datetime.now().isoformat(), "event": event, **data}
    _log_file.write(json.dumps(entry, default=str) + "\n")
    _log_file.flush()


# ── Plan mode ─────────────────────────────────────────────────────

_plan_mode = False

# Tools allowed in plan mode (read-only — no write_file or edit_file)
PLAN_TOOLS = [t for t in TOOL_DEFINITIONS if t["name"] not in ("write_file", "edit_file")]

PLAN_SYSTEM_ADDENDUM = """

--- PLAN MODE ---
You are in plan mode. Your job is to explore and understand, not to implement yet.
- Do NOT create, modify, or delete any files.
- Use read_file, list_files, bash (read-only commands: ls, cat, grep, find), web_search, and browse_url to gather context.
- Ask the user clarifying questions if anything is unclear.
- When ready, present a clear numbered implementation plan.
- The user will type /execute when they are ready to proceed.
"""

console = Console()

# ── Background process tracker ────────────────────────────────────

_bg_processes: list[dict] = []  # [{pid, cmd, proc, started}]


def _track_bg(proc: subprocess.Popen, cmd: str):
    _bg_processes.append({
        "pid": proc.pid,
        "cmd": cmd[:60],
        "proc": proc,
        "started": time.time(),
    })


def _reap_bg():
    """Remove finished processes from the tracker."""
    for p in _bg_processes[:]:
        if p["proc"].poll() is not None:
            _bg_processes.remove(p)


def _bg_status() -> str:
    """Return a short status string for background processes."""
    _reap_bg()
    if not _bg_processes:
        return ""
    names = ", ".join(p["cmd"][:25] for p in _bg_processes)
    return f" · [magenta]{len(_bg_processes)} bg[/] ({names})"

# ── Config ─────────────────────────────────────────────────────────

MODEL = os.environ.get("QWEN_MODEL", "gemma4:26b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
NUM_CTX = int(os.environ.get("QWEN_NUM_CTX", "32768"))
MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "8192"))
MAX_TURNS = int(os.environ.get("QWEN_MAX_TURNS", "50"))
COMPACT_PCT = float(os.environ.get("QWEN_COMPACT_PCT", "0.85"))
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



# ── Compaction ─────────────────────────────────────────────────────


async def compact_history(client: OllamaClient, messages: list, force: bool = False, ctx_used: int = 0) -> list:
    """Summarize older messages, keep recent ones intact. Returns new message list."""
    if not force and (ctx_used / NUM_CTX) < COMPACT_PCT:
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

    slog("compaction", old_messages=len(old), kept=len(recent), summary_len=len(summary))
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

# Commands that need confirmation and can optionally run in background
_RUN_PREFIXES = (
    "pip install", "pip3 install", "npm install", "yarn add", "brew install",
    "python ", "python3 ", "node ", "deno ", "bun run",
    "flask run", "uvicorn ", "gunicorn ", "django", "manage.py",
    "npm start", "npm run", "yarn start", "yarn dev",
    "cargo run", "go run", "java ", "ruby ", "php ",
    "docker ", "docker-compose", "make ",
)


def _needs_confirmation(tool_name: str, args: dict) -> tuple[str, bool] | None:
    """Return (description, can_background) if confirmation needed, else None."""
    if tool_name == "write_file":
        return f"Write to [cyan]{args.get('path', '?')}[/]", False
    if tool_name == "edit_file":
        return f"Edit [cyan]{args.get('path', '?')}[/]", False
    if tool_name == "bash":
        cmd = args.get("command", "")
        stripped = cmd.strip()
        # Check run commands first (can background)
        for prefix in _RUN_PREFIXES:
            if stripped.startswith(prefix) or f"&& {prefix}" in cmd or f"; {prefix}" in cmd:
                return f"Run: [yellow]{cmd}[/]", True
        # Then dangerous commands (no background option)
        for prefix in _DANGEROUS_PREFIXES:
            if stripped.startswith(prefix) or f"&& {prefix}" in cmd or f"; {prefix}" in cmd:
                return f"Run: [yellow]{cmd}[/]", False
    return None


def _read_single_key() -> str:
    """Read a single keypress without waiting for Enter."""
    import termios, tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _preview_tool(name: str, args: dict):
    """Print a preview of write_file or edit_file content before confirmation."""
    if name == "write_file":
        path = args.get("path", "")
        content = args.get("content", "")
        ext = os.path.splitext(path)[1].lstrip(".") or "text"
        lines = content.splitlines()
        preview = "\n".join(lines[:40])
        if len(lines) > 40:
            preview += f"\n[dim]... {len(lines) - 40} more lines[/]"
        console.print(Syntax(preview, ext, theme="monokai", line_numbers=True))

    elif name == "edit_file":
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        for line in old.splitlines():
            console.print(Text(f"  - {line}", style="red"))
        for line in new.splitlines():
            console.print(Text(f"  + {line}", style="green"))


async def _confirm(description: str, can_background: bool = False) -> str:
    """Ask user to confirm. Returns 'yes', 'bg', or 'no'. Single keypress."""
    if can_background:
        console.print(f"  [bold yellow]?[/] {description} [dim](y/b/n)[/] ", end="")
    else:
        console.print(f"  [bold yellow]?[/] {description} [dim](y/n)[/] ", end="")
    try:
        key = await asyncio.to_thread(_read_single_key)
        if key.lower() in ("y", "1"):
            console.print("[green]yes[/]")
            return "yes"
        if can_background and key.lower() in ("b", "2"):
            console.print("[magenta]background[/]")
            return "bg"
        console.print("[red]no[/]")
        return "no"
    except (EOFError, KeyboardInterrupt):
        console.print("[red]no[/]")
        return "no"


# ── Agent loop ─────────────────────────────────────────────────────


async def agent_loop(client: OllamaClient, messages: list, user_input: str, plan_mode: bool = False) -> tuple[str, int]:
    """Run the agentic tool-use loop. Returns (final_text, last_prompt_tokens)."""
    slog("user_input", text=user_input[:500])
    messages.append({"role": "user", "content": user_input})
    last_tokens = 0

    system = SYSTEM_PROMPT + (PLAN_SYSTEM_ADDENDUM if plan_mode else "")
    tools = PLAN_TOOLS if plan_mode else TOOL_DEFINITIONS

    for turn in range(MAX_TURNS):
        # Streaming: show thinking snippets and content tokens live
        import threading
        cancel_event = threading.Event()
        status = console.status(
            f"[bold blue]Thinking... [dim]Ctrl+C to cancel[/][/]",
            spinner="dots",
        )
        status.start()
        streaming_content = False
        _think_buf = []  # accumulate thinking text

        def _on_thinking(snippet):
            _think_buf.append(snippet)
            full = "".join(_think_buf)
            # Extract complete sentences: split on ". " followed by uppercase
            import re
            sentences = re.split(r'(?<=\. )(?=[A-Z])', full.replace("\n", " ").strip())
            # Show last 2 complete sentences (skip the last fragment if no trailing period)
            complete = [s.strip() for s in sentences if s.strip().endswith(".")]
            if not complete:
                # No complete sentence yet — show buffered text truncated
                s = full[-80:].replace("[", "(").replace("]", ")")
                status.update(f"[bold blue]Thinking:[/] [dim]{s}[/]")
                return
            display = complete[-2:] if len(complete) >= 2 else complete[-1:]
            s = " ".join(display)[-120:].replace("[", "(").replace("]", ")")
            status.update(f"[bold blue]Thinking:[/] [dim]{s}[/]")

        def _on_content(token):
            nonlocal streaming_content
            if not streaming_content:
                streaming_content = True
                status.stop()
                console.print()
            console.file.write(token)
            console.file.flush()

        try:
            response = await client.chat(
                system=system,
                messages=messages,
                tools=tools,
                on_thinking=_on_thinking,
                on_content=_on_content,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            cancel_event.set()  # signal the background thread to stop
            raise
        finally:
            status.stop()

        if streaming_content:
            console.file.write("\n")
            console.file.flush()

        last_tokens = response.prompt_tokens
        slog("llm_response", turn=turn, stop_reason=response.stop_reason,
             prompt_tokens=response.prompt_tokens,
             content_blocks=len(response.content),
             text_preview="\n".join(b.text[:200] for b in response.content if isinstance(b, TextBlock))[:400])

        if response.stop_reason == "end_turn":
            text = "\n".join(b.text for b in response.content if isinstance(b, TextBlock))
            messages.append({"role": "assistant", "content": text})
            # Don't re-print if we already streamed it
            if streaming_content:
                return "", last_tokens
            return text, last_tokens

        tool_results = []
        for block in response.content:
            if isinstance(block, TextBlock) and block.text and not streaming_content:
                console.print(Markdown(block.text))

            if isinstance(block, ToolUse):
                args_display = _format_tool_args(block.name, block.input)
                console.print(
                    f"  [dim]>[/] [bold purple]{block.name}[/] {args_display}"
                )
                slog("tool_call", tool=block.name, args={k: str(v)[:1000] for k, v in block.input.items()})

                # Detect bash commands with trailing & — run via our tracker
                cmd = block.input.get("command", "") if block.name == "bash" else ""
                if block.name == "bash" and cmd.rstrip().endswith("&"):
                    clean_cmd = cmd.rstrip().rstrip("&").strip()
                    proc = subprocess.Popen(
                        clean_cmd, shell=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        cwd=os.getcwd(),
                    )
                    _track_bg(proc, clean_cmd)
                    result = f"Launched in background (PID {proc.pid})."
                else:
                    # Ask for confirmation on writes/edits/dangerous/run commands
                    confirm_info = _needs_confirmation(block.name, block.input)
                    if confirm_info:
                        desc, can_bg = confirm_info
                        if block.name in ("write_file", "edit_file"):
                            _preview_tool(block.name, block.input)
                        choice = await _confirm(desc, can_bg)
                        if choice == "no":
                            result = "User denied this action."
                        elif choice == "bg":
                            proc = subprocess.Popen(
                                cmd, shell=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                cwd=os.getcwd(),
                            )
                            _track_bg(proc, cmd)
                            result = f"Launched in background (PID {proc.pid})."
                        else:
                            try:
                                result = await execute_tool(block.name, block.input)
                            except Exception as e:
                                result = f"Tool error: {e}"
                    else:
                        try:
                            result = await execute_tool(block.name, block.input)
                        except Exception as e:
                            result = f"Tool error: {e}"

                is_error = result.startswith("Error:")
                slog("tool_result", tool=block.name, error=is_error, result=result[:500])

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
            return text, last_tokens

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        # Mid-loop compaction: if context is filling up, compact before next LLM call
        if last_tokens > 0 and (last_tokens / NUM_CTX) >= COMPACT_PCT:
            old_len = len(messages)
            messages[:] = await compact_history(client, messages, ctx_used=last_tokens)
            if len(messages) < old_len:
                last_tokens = 0
                console.print(f"  [dim yellow]Auto-compacted mid-loop ({old_len} → {len(messages)} messages)[/]")

    return "Error: max turns reached.", last_tokens


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
    if name == "grep_file":
        return f"[cyan]{args.get('path', '')}[/] [dim]/{args.get('pattern', '')}/[/]"
    if name == "list_files":
        return f"[cyan]{args.get('pattern', '')}[/]"
    if name == "web_search":
        return f"[cyan]{args.get('query', '')}[/]"
    if name == "browse_url":
        return f"[cyan]{args.get('url', '')}[/]"
    if name == "analyze_image":
        return f"[cyan]{args.get('path', '')}[/] [dim]({args.get('question', 'describe')})[/]"
    return str(args)[:80]


# ── Help ──────────────────────────────────────────────────────────


def _show_help():
    console.print()
    console.print("[bold]Commands[/]")
    console.print("  [green]help[/], [green]/help[/]    Show this help")
    console.print("  [green]clear[/]           Reset conversation history")
    console.print("  [green]exit[/], [green]quit[/], [green]q[/]  Exit the agent")
    console.print("  [green]Ctrl+C[/]          Cancel a running request")
    console.print()
    console.print("[bold]Confirmations[/]")
    console.print("  [green]y[/]               Approve and run")
    console.print("  [green]b[/]               Approve and run in background [dim](servers, long tasks)[/]")
    console.print("  [dim]any other key[/]    Deny")
    console.print()
    console.print("[bold]Shortcuts[/] [dim](bypass the LLM)[/]")
    console.print("  [green]/ls[/] [dim][path][/]     List files in directory")
    console.print("  [green]/pwd[/]             Show current working directory")
    console.print("  [green]/cd[/] [dim]<path>[/]      Change working directory")
    console.print("  [green]/cat[/] [dim]<file>[/]     Print file contents")
    console.print("  [green]/model[/]            Pick a model [dim](shows installed list)[/]")
    console.print("  [green]/model[/] [dim]<name>[/]    Switch to a specific model directly")
    console.print("  [green]/compact[/]         Force conversation compaction now")
    console.print("  [green]/save[/] [dim]<file>[/]    Save last response to a file")
    console.print("  [green]/ps[/]              List background processes")
    console.print("  [green]/kill[/] [dim]<pid>[/]     Kill a background process")
    console.print()
    console.print("[bold]Plan mode[/]")
    console.print("  [green]Shift+Tab[/]        Toggle plan mode on/off")
    console.print("  [green]/plan[/]            Enter plan mode [dim](explore, no writes)[/]")
    console.print("  [green]/execute[/]         Exit plan mode, proceed with implementation")
    console.print()
    console.print("[bold]Tools available[/]")
    console.print("  [purple]bash[/]            Run shell commands (git, tests, builds)")
    console.print("  [purple]read_file[/]       Read file contents with line numbers")
    console.print("  [purple]write_file[/]      Create or overwrite files [dim](confirms)[/]")
    console.print("  [purple]edit_file[/]       Find-and-replace in files [dim](confirms)[/]")
    console.print("  [purple]list_files[/]      Glob pattern file search")
    console.print("  [purple]web_search[/]      Search the web via DuckDuckGo")
    console.print("  [purple]browse_url[/]      Fetch and read a web page")
    console.print("  [purple]analyze_image[/]   Describe a screenshot or image")
    console.print()
    console.print("[bold]Configuration[/] [dim](env vars)[/]")
    console.print(f"  QWEN_MODEL          [dim]{rich_escape(MODEL)}[/]")
    console.print(f"  QWEN_NUM_CTX        [dim]{NUM_CTX}[/]")
    console.print(f"  QWEN_MAX_TOKENS     [dim]{MAX_TOKENS}[/]")
    console.print(f"  QWEN_MAX_TURNS      [dim]{MAX_TURNS}[/]")
    console.print()


# ── Slash shortcuts ───────────────────────────────────────────────


async def _handle_slash(user_input: str) -> bool:
    """Handle slash commands. Returns True if handled."""
    parts = user_input.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/ls":
        path = arg or "."
        import subprocess
        result = subprocess.run(["ls", "-la", path], capture_output=True, text=True)
        console.print(result.stdout or result.stderr)
        return True

    if cmd == "/pwd":
        console.print(os.getcwd())
        return True

    if cmd == "/cd":
        if not arg:
            console.print("[red]Usage: /cd <path>[/]")
        else:
            try:
                os.chdir(os.path.expanduser(arg))
                console.print(f"[dim]{os.getcwd()}[/]")
            except Exception as e:
                console.print(f"[red]{e}[/]")
        return True

    if cmd == "/cat":
        if not arg:
            console.print("[red]Usage: /cat <file>[/]")
        else:
            try:
                with open(os.path.expanduser(arg)) as f:
                    console.print(f.read())
            except Exception as e:
                console.print(f"[red]{e}[/]")
        return True

    # /model is handled separately in _handle_slash after this block

    if cmd == "/compact":
        return "compact"  # special signal handled in REPL loop

    if cmd == "/ps":
        _reap_bg()
        if not _bg_processes:
            console.print("[dim]No background processes.[/]")
        else:
            for p in _bg_processes:
                elapsed = int(time.time() - p["started"])
                mins, secs = divmod(elapsed, 60)
                console.print(
                    f"  PID [cyan]{p['pid']}[/]  "
                    f"[dim]{mins}m{secs:02d}s[/]  {p['cmd']}"
                )
        return True

    if cmd == "/kill":
        if not arg:
            console.print("[red]Usage: /kill <pid>[/]")
        else:
            try:
                pid = int(arg)
                target = next((p for p in _bg_processes if p["pid"] == pid), None)
                if target:
                    target["proc"].terminate()
                    _bg_processes.remove(target)
                    console.print(f"[dim]Killed PID {pid}[/]")
                else:
                    console.print(f"[red]PID {pid} not found in background processes[/]")
            except ValueError:
                console.print("[red]Usage: /kill <pid>[/]")
        return True

    if cmd == "/save":
        return f"save:{arg}"

    if cmd == "/plan":
        return "plan_on"

    if cmd == "/execute":
        return "plan_off"

    if cmd == "/model":
        if arg:
            return f"switch_model:{arg}"
        # No arg — show picker
        chosen = await _model_picker(current=MODEL)
        if chosen:
            return f"switch_model:{chosen}"
        return True

    return False  # not a known slash command — pass to LLM


# ── Model picker ─────────────────────────────────────────────────


async def fetch_models() -> list[str]:
    """Return list of model names from Ollama."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_HOST}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                data = await resp.json()
        return sorted(m["name"] for m in data.get("models", []))
    except Exception:
        return []


async def _model_picker(current: str) -> str | None:
    """Show a numbered list of installed models. Returns chosen name or None."""
    models = await fetch_models()
    if not models:
        console.print("[red]Could not reach Ollama — is it running?[/]")
        return None

    console.print()
    console.print("[bold]Installed models[/]")
    for i, name in enumerate(models, 1):
        marker = " [bold green]◀ current[/]" if name == current else ""
        console.print(f"  [cyan]{i}[/]  {rich_escape(name)}{marker}", highlight=False)
    console.print()
    console.print(f"  [dim]Enter number to switch, or press Enter to keep [cyan]{rich_escape(current)}[/][/]")

    try:
        choice = await asyncio.to_thread(input, "  > ")
        choice = choice.strip()
        if not choice:
            return None
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
        console.print("[red]Invalid choice.[/]")
        return None
    except (ValueError, EOFError, KeyboardInterrupt):
        return None


# ── REPL ───────────────────────────────────────────────────────────


async def main():
    global _active_task, _plan_mode, MODEL

    loop = asyncio.get_running_loop()
    _install_signal_handlers(loop)
    slog("session_start", model=MODEL, num_ctx=NUM_CTX, cwd=os.getcwd(), log_file=_log_path)

    # Model picker on startup (skip if QWEN_MODEL env var is explicitly set)
    if not os.environ.get("QWEN_MODEL"):
        chosen = await _model_picker(current=MODEL)
        if chosen:
            MODEL = chosen

    console.print(
        Panel(
            f"[bold]Vaib Kodar[/]\n"
            f"Model: [cyan]{rich_escape(MODEL)}[/]  Ctx: [cyan]{NUM_CTX}[/]\n"
            f"[dim]exit · clear · Ctrl+C to cancel · Shift+Tab for plan mode[/]\n"
            f"[dim]Log: {_log_path}[/]",
            border_style="blue",
            width=80,
            highlight=False,
        )
    )

    client = OllamaClient(
        host=OLLAMA_HOST,
        model=MODEL,
        num_ctx=NUM_CTX,
        max_tokens=MAX_TOKENS,
    )

    messages = []
    last_prompt_tokens = 0

    history_path = os.path.expanduser("~/.qwen-agent-history")

    yellow_style = Style.from_dict({'': 'ansiyellow'})

    kb = KeyBindings()

    @kb.add('s-tab')
    def _toggle_plan(event):
        global _plan_mode
        _plan_mode = not _plan_mode
        event.app.invalidate()

    def _prompt_text():
        if _plan_mode:
            return "[PLAN] Human: "
        return "Human: "

    session = PromptSession(
        history=FileHistory(history_path),
        style=yellow_style,
        key_bindings=kb,
    )

    while True:
        try:
            # Status line above the input
            bg = _bg_status()
            plan_indicator = " · [bold green]PLAN MODE[/]" if _plan_mode else ""
            model_short = rich_escape(MODEL.split(":")[0] if ":" in MODEL else MODEL)
            ctx_pct = f" ({last_prompt_tokens*100//NUM_CTX}%)" if last_prompt_tokens > 0 else ""
            ctx_part = f" · ctx {last_prompt_tokens:,}/{NUM_CTX:,}{ctx_pct}" if last_prompt_tokens > 0 else ""
            status = f"[dim]{model_short} · turns {len(messages)}{ctx_part}{bg}[/]{plan_indicator}"

            console.print()
            console.print(status, highlight=False)
            console.print(Rule(style="dim cyan"))
            user_input = await asyncio.to_thread(session.prompt, _prompt_text, multiline=False)
            console.print(Rule(style="dim cyan"))
            console.print()
        except EOFError:
            console.print("\n[dim]Bye![/]")
            break
        except KeyboardInterrupt:
            continue

        user_input = user_input.strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Bye![/]")
            break
        if user_input.lower() == "clear":
            messages.clear()
            last_prompt_tokens = 0
            console.print("[dim]Conversation cleared.[/]")
            continue
        if user_input.lower() in ("help", "/help"):
            _show_help()
            continue

        # Slash shortcuts — bypass the LLM
        if user_input.startswith("/"):
            handled = await _handle_slash(user_input)
            if handled == "compact":
                if len(messages) < 4:
                    console.print("[dim]Not enough history to compact.[/]")
                else:
                    messages[:] = await compact_history(client, messages, force=True)
                    last_prompt_tokens = 0
                continue
            if isinstance(handled, str) and handled.startswith("save:"):
                filename = handled.split(":", 1)[1].strip()
                if not filename:
                    console.print("[red]Usage: /save <filename>[/]")
                else:
                    # Find last assistant message
                    last_response = None
                    for msg in reversed(messages):
                        if msg["role"] == "assistant" and isinstance(msg.get("content"), str) and msg["content"]:
                            last_response = msg["content"]
                            break
                    if last_response:
                        path = os.path.expanduser(filename)
                        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                        with open(path, "w") as f:
                            f.write(last_response)
                        console.print(f"[dim]Saved to {path}[/]")
                    else:
                        console.print("[red]No assistant response to save.[/]")
                continue
            if handled == "plan_on":
                _plan_mode = True
                console.print("[bold green]Plan mode ON[/] — exploring only, no writes. Type /execute to proceed.")
                continue
            if handled == "plan_off":
                _plan_mode = False
                console.print("[dim]Plan mode OFF — back to normal.[/]")
                continue
            if isinstance(handled, str) and handled.startswith("switch_model:"):
                new_model = handled.split(":", 1)[1].strip()
                MODEL = new_model
                client.model = new_model
                messages.clear()
                last_prompt_tokens = 0
                console.print(f"[bold green]Switched to [cyan]{rich_escape(new_model)}[/]. Conversation cleared.[/]")
                continue
            if handled:
                continue

        # Run agent — Ctrl+C here cancels the task, not the app
        old_len = len(messages)
        messages[:] = await compact_history(client, messages, ctx_used=last_prompt_tokens)
        if len(messages) < old_len:
            last_prompt_tokens = 0

        _active_task = asyncio.current_task()
        try:
            result, last_prompt_tokens = await agent_loop(client, messages, user_input, plan_mode=_plan_mode)
            if result:
                console.print()
                console.print(Markdown(result))
        except asyncio.CancelledError:
            _cleanup_messages(messages, user_input)
            console.print("\n[yellow]Cancelled.[/]")
        except Exception as e:
            from rich.text import Text
            console.print(Text(f"\nError: {e}", style="red"))
        finally:
            _active_task = None


if __name__ == "__main__":
    asyncio.run(main())
