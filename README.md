# Qwen Coding Agent

A terminal-based coding assistant powered by a local Qwen 3.5 35B model via Ollama. Zero API costs, fully private, runs entirely on your machine.

Built as a lightweight alternative to cloud-based coding agents — same core workflow (read, edit, run, iterate) without sending your code anywhere.

## What It Does

You type a request, the agent uses tools to accomplish it:

```
Human: find all TODO comments in the project and list them
  > bash grep -rn "TODO" src/
    src/auth.py:42:    # TODO: add rate limiting
    src/db.py:18:     # TODO: connection pooling

Found 2 TODO comments across the project.
```

The agent can chain multiple tools per request — read a file, edit it, run tests, fix failures — up to 20 turns per interaction.

## Tools

| Tool | What It Does |
|------|-------------|
| `bash` | Run shell commands (git, tests, builds, etc.) — 120s timeout |
| `read_file` | Read file contents with line numbers |
| `write_file` | Create or overwrite files |
| `edit_file` | Surgical string replacement (must read first) |
| `list_files` | Glob pattern file search |
| `web_search` | DuckDuckGo search |
| `browse_url` | Fetch and extract text from web pages |
| `analyze_image` | Describe screenshots/images using vision |

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) with Qwen 3.5 35B pulled.

```bash
# Pull the model (if you haven't already)
ollama pull qwen3.5:35b-a3b

# Clone and install
git clone git@github.com:gtgabriel/coding_agent.git
cd coding_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Run from any project directory
cd /path/to/your/project
/path/to/coding_agent/.venv/bin/python3 /path/to/coding_agent/agent.py
```

Or add a shell alias to `~/.zshrc`:

```bash
alias qwen_agent='PYTHONPATH="/path/to/coding_agent" /path/to/coding_agent/.venv/bin/python3 /path/to/coding_agent/agent.py'
```

Then just:

```bash
cd my-project
qwen_agent
```

### Commands

- `exit` / `quit` / `q` — leave the agent
- `clear` — reset conversation history
- `Ctrl+C` — cancel a running request (doesn't kill the app)

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_MODEL` | `qwen3.5:35b-a3b` | Ollama model name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `QWEN_NUM_CTX` | `8192` | Context window size |
| `QWEN_MAX_TOKENS` | `4096` | Max tokens per response |
| `QWEN_MAX_TURNS` | `20` | Max tool-use turns per request |
| `QWEN_COMPACT_THRESHOLD` | `20` | Messages before auto-compaction |
| `QWEN_COMPACT_KEEP` | `6` | Recent messages kept during compaction |

Example with a different model:

```bash
QWEN_MODEL=gemma3:27b qwen_agent
```

## Architecture

```
agent.py      REPL loop + agent loop + tool confirmation
llm.py        Ollama client (native /api/chat with tool calling)
tools.py      8 tools — definitions + implementations
prompts.py    System prompt
```

The agent loop follows the standard pattern: prompt model with tools → model returns tool calls → execute tools → feed results back → repeat until done.

Conversation history auto-compacts when it gets long — older messages are summarized by the model, recent ones kept intact.

### Safety

Write operations (`write_file`, `edit_file`) and dangerous bash commands (`rm`, `sudo`, `git push`, etc.) require explicit `y/N` confirmation before executing.

## Dependencies

- `aiohttp` — async HTTP client for Ollama API
- `rich` — terminal formatting and markdown rendering
- `prompt_toolkit` — input with history and keybindings
