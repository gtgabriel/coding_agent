# Vaib Kodar

A terminal-based coding agent powered by local LLMs via Ollama. Zero API costs, fully private, runs entirely on your machine.

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

The agent chains multiple tools per request — read a file, edit it, run tests, fix failures — up to 50 turns per interaction.

## Tools

| Tool | What It Does |
|------|-------------|
| `bash` | Run shell commands (git, tests, builds, etc.) — 120s timeout |
| `read_file` | Read file contents with line numbers |
| `write_file` | Create or overwrite files |
| `edit_file` | Surgical string replacement (must read first) |
| `list_files` | Glob pattern file search |
| `grep_file` | Search file contents for patterns |
| `web_search` | DuckDuckGo search |
| `browse_url` | Fetch and extract text from web pages |
| `analyze_image` | Describe screenshots/images using vision |

## Agent Features

### Thinking Display
The agent shows real-time thinking snippets as the model reasons through problems. A spinner displays during inference, and thinking summaries appear between tool calls with elapsed time.

### File Read Deduplication
The agent tracks which files have been read and on which turn. If the model tries to re-read an unmodified file, it returns a reminder that the content is already in the conversation context — saving tokens and time. Edits mark files as stale so the next read goes through. Tracking resets on conversation compaction or `clear`.

### Safety Guards
- **Write confirmation**: `write_file`, `edit_file`, and dangerous bash commands (`rm`, `sudo`, `git push`, etc.) require explicit `y/n` confirmation before executing.
- **Denial handling**: If you deny an action, all remaining tool calls in that turn are skipped and the agent asks what you'd like to do instead.
- **Stdin buffer flush**: Confirmation prompts drain any buffered keypresses before reading, preventing accidental approvals or denials.

### Context Management
- **Adaptive compaction**: When the context window fills to 85%, older messages are summarized by the model and replaced with a compact summary, keeping recent messages intact.
- **Working memory**: The agent tracks file modifications across the session and surfaces them when relevant (e.g., after a bash error).
- **Session logging**: Every session is logged to `~/.vaib-kodar/logs/` as JSONL with timestamps, tool calls, results, and thinking metrics.

### Plan Mode
Press `Shift+Tab` or type `/plan` to enter plan mode — the agent can explore and read files but cannot make any writes. Useful for understanding a codebase before committing to changes.

## Recommended Models

Tested on Apple Silicon (M-series Macs). These models use quantisation formats optimised for Metal/unified memory:

| Model | Size | Best For | Pull Command |
|-------|------|----------|-------------|
| `qwen3.5:35b-a3b-nvfp4` | 21 GB | Best overall — strong tool use, thinking, editing | `ollama pull qwen3.5:35b-a3b-nvfp4` |
| `gemma4:26b` | 17 GB | Good reasoning, fast inference, occasional thinking loops | `ollama pull gemma4:26b` |
| `qwen3.5:9b-nvfp4` | 8.9 GB | Lightweight — fits 8GB+ machines, good for simple tasks | `ollama pull qwen3.5:9b-nvfp4` |

**Notes:**
- Qwen 3.5 35B uses a Mixture-of-Experts (MoE) architecture — only ~3B parameters active per token, so it runs much faster than a dense 35B model despite the larger download.
- Gemma 4 26B is also MoE — fast inference with strong reasoning capabilities.
- All models support Ollama's native tool calling and thinking modes.
- A Mac with 32GB+ unified memory is recommended for the 35B/26B models. The 9B model runs comfortably on 16GB.

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) with a supported model pulled.

```bash
# Pull recommended model
ollama pull qwen3.5:35b-a3b-nvfp4
# or lighter alternatives
ollama pull gemma4:26b
ollama pull qwen3.5:9b-nvfp4

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
alias qa='/path/to/coding_agent/.venv/bin/python3 /path/to/coding_agent/agent.py'
```

Then just:

```bash
cd my-project
qa
```

On startup, a model picker shows installed Ollama models. Select one or press Enter to keep the default.

### Commands

| Command | Action |
|---------|--------|
| `exit` / `quit` / `q` | Leave the agent |
| `clear` | Reset conversation history |
| `help` / `/help` | Show help and configuration |
| `/model` | Switch model mid-session |
| `/model <name>` | Switch to a specific model |
| `/compact` | Force conversation compaction |
| `/save <file>` | Save last response to a file |
| `/plan` | Toggle plan mode (read-only) |
| `Ctrl+C` | Cancel a running request |

## Configuration

All settings via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VK_MODEL` | `gemma4:26b` | Ollama model name (skips picker if set) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server URL |
| `VK_NUM_CTX` | `32768` | Context window size |
| `VK_MAX_TOKENS` | `8192` | Max tokens per response |
| `VK_MAX_TURNS` | `50` | Max tool-use turns per request |
| `VK_COMPACT_PCT` | `0.85` | Context usage % that triggers compaction |
| `VK_COMPACT_KEEP` | `6` | Recent messages kept during compaction |

Example with a different model and context size:

```bash
VK_MODEL=qwen3.5:9b-nvfp4 VK_NUM_CTX=16384 qa
```

## Architecture

```
agent.py      REPL loop, agent loop, tool confirmation, thinking display,
              file read tracking, context compaction, session logging
llm.py        Ollama client (streaming /api/chat with thinking tag parsing)
tools.py      9 tools — definitions + implementations
prompts.py    System prompt
```

The agent loop follows the standard pattern: prompt model with tools → model returns tool calls → confirm if needed → execute tools → feed results back → repeat until done.

Conversation history auto-compacts when context fills up — older messages are summarized by the model, recent ones kept intact. File read tracking prevents redundant reads within a session.

### Thinking Mode

The agent supports model thinking/reasoning via two paths:
- **Native thinking channel**: When the model supports Ollama's `think: true` parameter, thinking tokens stream through a dedicated channel.
- **Embedded tag parsing**: For models that embed thinking in `<think>...</think>` or `<|channel>...<channel|>` tags within content, the agent parses and extracts them automatically.

Thinking summaries display between tool calls with elapsed time, providing visibility into the model's reasoning without cluttering the output.

## Dependencies

- `aiohttp` — async HTTP client for Ollama API
- `rich` — terminal formatting and markdown rendering
- `prompt_toolkit` — input with history and keybindings
