"""Coding tools — bash, read, write, edit, list files, web search, browse, and vision."""

import os
import re
import ssl
import subprocess
import glob as globmod
import base64
import json
import aiohttp
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import quote

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Configuration ──────────────────────────────────────────────────

MAX_OUTPUT = 8000  # chars
MODEL = os.environ.get("QWEN_MODEL", "qwen3.5:35b-a3b-nvfp4")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")

# ── Tool definitions (Anthropic format) ───────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "bash",
        "description": "Execute a shell command and return stdout/stderr. Use for git, tests, builds, installs, etc. Timeout: 120s.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file's contents. Returns lines with line numbers. Always read before editing.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
                "offset": {"type": "integer", "description": "Start line (1-based, optional)"},
                "limit": {"type": "integer", "description": "Number of lines to read (optional)"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file. You MUST read the file first. The old_string must match the file content exactly — do NOT include line numbers or tab prefixes from read_file output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {"type": "string", "description": "Exact string to find and replace"},
                "new_string": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "list_files",
        "description": "Find files matching a glob pattern. Returns matching file paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern to match files"},
                "path": {"type": "string", "description": "Directory to search in (default: current dir)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_file",
        "description": "Search for a pattern in a file. Returns matching lines with line numbers. Use before read_file to find the right section.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to search"},
                "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
            },
            "required": ["path", "pattern"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo. Returns top 5 results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "browse_url",
        "description": "Fetch and read the text content of a web page URL. Returns extracted text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full URL to fetch"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "analyze_image",
        "description": "Read an image file (PNG, JPG, etc.) and describe its contents using vision capabilities. Use for screenshots or diagrams.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the image file"},
                "question": {"type": "string", "description": "Specific question about the image (optional)"},
            },
            "required": ["path"],
        },
    },
]

# ── Tool implementations ───────────────────────────────────────────

async def bash(command: str) -> str:
    """Execute a shell command."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=os.getcwd(),
        )
        output = result.stdout + result.stderr
        if not output:
            output = f"(exit code {result.returncode})"
        return output[:MAX_OUTPUT] + ("..." if len(output) > MAX_OUTPUT else "")
    except subprocess.TimeoutExpired:
        return "Error: command timed out (120s limit)"
    except Exception as e:
        return f"Error: {e}"

async def read_file(path: str, offset: int = None, limit: int = None) -> str:
    """Read a file with line numbers."""
    try:
        path = os.path.expanduser(path)
        with open(path, "r") as f:
            lines = f.readlines()
        start = max(0, (offset or 1) - 1)
        end = start + limit if limit else len(lines)
        numbered = [f"{i}\t{l.rstrip()}" for i, l in enumerate(lines[start:end], start=start + 1)]
        output = "\n".join(numbered)
        return output[:MAX_OUTPUT] + ("..." if len(output) > MAX_OUTPUT else "")
    except Exception as e:
        return f"Error: {e}"

async def write_file(path: str, content: str) -> str:
    """Create or overwrite a file."""
    try:
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        lines = content.count("\n") + 1
        return f"Wrote {lines} lines to {path}"
    except Exception as e:
        return f"Error: {e}"

_edit_fail_count: dict = {}  # track consecutive failures per file

async def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace exact string in a file."""
    import re
    try:
        path = os.path.expanduser(path)

        # Guard: no-op edit
        if old_string == new_string:
            return "Error: old_string and new_string are identical. No change needed."

        # Guard: truncated strings (model hit output limit mid-argument)
        for label, s in [("old_string", old_string), ("new_string", new_string)]:
            if s and len(s) > 50:
                stripped = s.rstrip()
                # Ends mid-word (letter/digit not followed by punctuation or newline)
                if stripped and stripped[-1].isalnum() and not stripped.endswith(("true", "false", "null", "none", "None")):
                    last_line = stripped.rsplit("\n", 1)[-1].rstrip()
                    # Check if it looks like a cut-off line (not a complete statement)
                    if not any(last_line.endswith(e) for e in ("}", ";", ")", "]", ">", ",", ":", "{", "(", "=", '"', "'")):
                        return f"Error: {label} appears truncated (ends with: ...{stripped[-50:]!r}). Use a smaller edit scope — target fewer lines."

        # Guard: force re-read after 2 consecutive failures on same file
        if _edit_fail_count.get(path, 0) >= 2:
            _edit_fail_count[path] = 0
            return "Error: multiple edit failures on this file. You MUST read_file first to see the current content before trying again."

        with open(path, "r") as f:
            content = f.read()

        candidate = old_string

        # Fix 1: Strip line numbers from read_file output (e.g. "42\t" prefix)
        if candidate not in content:
            stripped = re.sub(r"^\d+\t", "", candidate, flags=re.MULTILINE)
            if stripped != candidate and stripped in content:
                candidate = stripped

        # Fix 2: Normalize trailing whitespace per line
        if candidate not in content:
            normalized = "\n".join(l.rstrip() for l in candidate.split("\n"))
            if normalized in content:
                candidate = normalized

        # Fix 3: Tab/space mismatch — try both directions
        if candidate not in content:
            tab_to_spaces = candidate.replace("\t", "    ")
            spaces_to_tab = re.sub(r"^(    )+", lambda m: "\t" * (len(m.group()) // 4), candidate, flags=re.MULTILINE)
            if tab_to_spaces in content:
                candidate = tab_to_spaces
            elif spaces_to_tab in content:
                candidate = spaces_to_tab

        # Fix 4: Fuzzy indent match — find the block by stripped content, use file's actual indentation
        if candidate not in content:
            candidate_lines = [l.rstrip() for l in candidate.split("\n")]
            content_lines = content.split("\n")
            stripped_candidate = [l.lstrip() for l in candidate_lines]
            # Slide through file looking for a match on stripped content
            for start in range(len(content_lines) - len(candidate_lines) + 1):
                window = content_lines[start:start + len(candidate_lines)]
                stripped_window = [l.lstrip() for l in window]
                if stripped_window == stripped_candidate:
                    # Found it — use the file's actual text
                    candidate = "\n".join(window)
                    break

        if candidate not in content:
            # Show a hint: find the closest matching line to help the model
            first_line = old_string.strip().split("\n")[0].strip()
            hints = []
            for i, line in enumerate(content.splitlines(), 1):
                if first_line and first_line in line:
                    hints.append(f"  line {i}: {line.rstrip()[:100]}")
            hint_text = "\n".join(hints[:3])
            if hint_text:
                _edit_fail_count[path] = _edit_fail_count.get(path, 0) + 1
                return f"Error: old_string not found (whitespace mismatch?). Similar lines in file:\n{hint_text}\nRead the file again to get exact content."
            _edit_fail_count[path] = _edit_fail_count.get(path, 0) + 1
            return "Error: old_string not found. Read the file again to ensure exact match. Do NOT include line numbers from read_file output."

        if content.count(candidate) > 1:
            return "Error: old_string is not unique. Provide more context."
        new_content = content.replace(candidate, new_string, 1)
        with open(path, "w") as f:
            f.write(new_content)
        _edit_fail_count.pop(path, None)  # reset on success
        return f"Edited {path} ({len(new_string.splitlines())} lines replaced)"
    except Exception as e:
        return f"Error: {e}"

async def grep_file(path: str, pattern: str) -> str:
    """Search for a pattern in a file, return matching lines with numbers."""
    try:
        path = os.path.expanduser(path)
        with open(path, "r") as f:
            lines = f.readlines()
        matches = []
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                matches.append(f"{i}\t{line.rstrip()}")
        if not matches:
            return f"No matches for '{pattern}' in {path}"
        output = "\n".join(matches[:50])
        return output[:MAX_OUTPUT]
    except Exception as e:
        return f"Error: {e}"

async def list_files(pattern: str, path: str = None) -> str:
    """Find files matching a glob pattern."""
    try:
        search_dir = os.path.expanduser(path or os.getcwd())
        matches = sorted(globmod.glob(os.path.join(search_dir, pattern), recursive=True))
        files = [os.path.relpath(m, search_dir) for m in matches if os.path.isfile(m)]
        return "\n".join(files[:100]) or "No files found."
    except Exception as e:
        return f"Error: {e}"

async def web_search(query: str) -> str:
    """Search DuckDuckGo."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req, timeout=15, context=SSL_CTX).read().decode()
        results = re.findall(r'class="result__a" href="([^"]*)">(.*?)</a>.*?snippet">(.*?)</a>', html, re.S)
        items = [f"**{t}**\n{s}\n{h}" for h, t, s in results[:5]]
        return "\n\n".join(items) or "No results found."
    except Exception as e:
        return f"Search error: {e}"

async def browse_url(url: str) -> str:
    """Fetch URL text content."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urlopen(req, timeout=20, context=SSL_CTX).read().decode("utf-8", errors="replace")
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", raw, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:4000]
    except Exception as e:
        return f"Browse error: {e}"

async def analyze_image(path: str, question: str = "Describe this image in detail.") -> str:
    """Vision tool for Qwen 3.5 via Ollama."""
    try:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: File {path} not found."

        with open(path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "/no_think\n" + question, "images": [b64_data]},
            ],
            "stream": False,
            "options": {"num_ctx": 4096, "num_predict": 1024},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_HOST}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),
            ) as resp:
                raw = await resp.read()
                data = json.loads(raw)
                if "error" in data:
                    return f"Vision error: {data['error']}"
                return data.get("message", {}).get("content", "No analysis returned.")
    except Exception as e:
        return f"Vision error: {e}"

# ── Dispatcher ─────────────────────────────────────────────────────

TOOL_HANDLERS = {
    "bash": lambda args: bash(args.get("command", "")),
    "read_file": lambda args: read_file(args.get("path", ""), args.get("offset"), args.get("limit")),
    "write_file": lambda args: write_file(args.get("path", ""), args.get("content", "")),
    "edit_file": lambda args: edit_file(args.get("path", ""), args.get("old_string", ""), args.get("new_string", "")),
    "grep_file": lambda args: grep_file(args.get("path", ""), args.get("pattern", "")),
    "list_files": lambda args: list_files(args.get("pattern", ""), args.get("path")),
    "web_search": lambda args: web_search(args.get("query", "")),
    "browse_url": lambda args: browse_url(args.get("url", "")),
    "analyze_image": lambda args: analyze_image(args.get("path", ""), args.get("question", "Describe this image.")),
}

async def execute_tool(name: str, input_data: dict) -> str:
    handler = TOOL_HANDLERS.get(name)
    return await handler(input_data) if handler else f"Unknown tool: {name}"