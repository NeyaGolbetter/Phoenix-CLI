# 🔥 PHOENIX — Rise. Chat. Create.

**AI that rises with you.** A provider-agnostic AI assistant for the terminal —
built to run beautifully on Termux, desktop Linux, macOS, and WSL.

Phoenix talks the **OpenAI-compatible API** format. Plug in *any* provider —
**Ollama, LM Studio, vLLM, llama.cpp, OpenRouter, Together AI, Groq, DeepSeek,
Mistral, Fireworks, xAI, LocalAI** — configure it once, and chat from anywhere,
including your Android phone. Supports **MCP (Model Context Protocol)** so your
AI can use tools like the **Roblox MCP server** to build games from the terminal.

```
$ phoenix                            # one command → interactive chat, just type
$ phoenix "write a python script"    # one-shot prompt
$ phoenix setup                      # configure provider + model (interactive picker)
$ phoenix models --select            # pick a model from a numbered list
$ phoenix mcp add-roblox             # one-command Roblox MCP setup
$ phoenix status                     # check configuration
```

Running bare `phoenix` drops you straight into chat — **no commands needed**.
Type messages freely until you type `/exit` (or press Ctrl+D). Slash commands
(`/help`, `/model`, `/save`, …) are entirely optional. You can even keep typing
while a reply is still streaming — input is buffered and sent in order, and
Ctrl+C cancels the in-flight reply.

---

## 📑 Table of contents

1. [Installation](#1-installation)
2. [Initial setup (the 3-step tutorial)](#2-initial-setup)
3. [Using models — the interactive picker](#3-using-models--the-interactive-picker)
4. [MCP setup — Roblox MCP on mobile](#4-mcp-setup--roblox-mcp-on-mobile)
5. [All 30 chat commands](#5-all-30-chat-commands)
6. [Single-prompt mode](#6-single-prompt-mode)
7. [Provider cheat sheet](#7-provider-cheat-sheet)
8. [Termux tips & troubleshooting](#8-termux-tips--troubleshooting)
9. [Project layout & development](#9-project-layout--development)

---

## 1. Installation

### Termux (Android)

Open Termux (get it from F-Droid, not Play Store) and run:

```bash
# 1. Update packages
pkg update && pkg upgrade

# 2. Install Python and git
pkg install python git

# 3. Clone Phoenix CLI
git clone https://github.com/NeyaGolbetter/Phoenix-CLI
cd Phoenix-CLI

# 4. Install (pure Python — no compiler needed)
pip install .

# 5. (Optional) line editing + syntax highlighting
pip install ".[repl,highlight]"
```

`phoenix` is now on your `$PATH` (`$PREFIX/bin`). If the shell doesn't see it,
run `hash -r` or use the fallback `python -m phoenix_cli`.

### Desktop (Linux / macOS / WSL)

Same steps, skip the `pkg` commands. Requires Python 3.9+.

```bash
git clone https://github.com/NeyaGolbetter/Phoenix-CLI
cd Phoenix-CLI
python3 -m venv .venv && source .venv/bin/activate
pip install ".[repl,highlight]"
```

### Optional extras (both pure Python — no compiler)

| Extra        | Package          | What it gives you                                   |
|-------------|------------------|-----------------------------------------------------|
| `repl`      | `prompt_toolkit` | cursor keys, history, proper line editing in chat   |
| `highlight` | `pygments`       | full syntax highlighting in markdown code blocks    |

Without them, chat still works (falls back to plain `input()`), and code blocks
render unhighlighted.

---

## 2. Initial setup

Run `phoenix setup`. It asks three questions (the API key is hidden — ● bullets
appear as you type so you know input is being received):

```
$ phoenix setup

Phoenix setup — the provider can be any OpenAI-compatible API.
Press Enter to keep the current value.

examples:
  Ollama (local)     http://localhost:11434
  LM Studio (local)  http://localhost:1234
  vLLM (local)       http://localhost:8000
  OpenRouter         https://openrouter.ai/api
  Together AI        https://api.together.xyz
  Groq               https://api.groq.com/openai
  Any custom server    myserver.example.com:8080
BASE_URL: myserver.example.com:8080
API_KEY (Enter for none — local servers usually need none): sk-...

Connecting to the provider to fetch models...
✓ found 42 model(s)

Select a model (enter its number):
 #     Model
 1     llama3.2
 2     qwen2.5-coder
 3     gpt-4o-mini  ✓
 4     deepseek-coder-v2
 ...
Enter number (0 to skip): 3
✓ selected: gpt-4o-mini

Enable MCP (Model Context Protocol) tools? (for Roblox MCP etc.) [y/N]: n

╭─ Saved ──────────────────────────────────────╮
│  BASE_URL     http://myserver.example.com:8080/v1  │
│  API_KEY      sk-****mini                          │
│  MODEL_NAME   gpt-4o-mini                          │
│  MCP          disabled                             │
╰──────────────────────────────────────────────────╯

Next: run `phoenix` to start chatting, or try phoenix "hello!" for a one-shot
```

### What happens under the hood

1. **BASE_URL** — the endpoint of your provider. If you type a bare hostname
   (`myserver.example.com:8080`) it gets `http://` prepended automatically.
   If it has no path, `/v1` is appended (`http://localhost:11434` →
   `http://localhost:11434/v1`). Explicit paths like
   `https://api.groq.com/openai` are kept as-is.

2. **API_KEY** — required for cloud providers, usually empty for local servers.
   Stored in `~/.phoenix_config.json` with file permissions `0600` (owner-only).

3. **MODEL_NAME** — Phoenix connects to your provider and fetches every model
   it advertises. You pick one from a numbered list. If the auto-fetch fails
   (no network, provider doesn't expose `/v1/models`), you can type the name
   manually.

4. **MCP toggle** — enables tool-use. When ON, Phoenix loads `~/.phoenix_mcp.json`
   (where your MCP servers are defined) and the AI can call tools during chat.

### Environment variables (alternative to the file)

```bash
export PHOENIX_BASE_URL="http://192.168.1.20:11434"   # e.g. Ollama on your PC
export PHOENIX_API_KEY="sk-..."
export PHOENIX_MODEL="llama3.2"
```

Env vars take priority over the file. `PHOENIX_CONFIG=/path/to/file.json`
points at an alternative config file (handy for multiple provider profiles).

---

## 3. Using models — the interactive picker

Phoenix gives you three ways to pick a model:

### Method 1 — `phoenix models --select` (quick pick)

```bash
$ phoenix models --select

3 model(s) available from http://localhost:11434/v1

 #     Model
 1     llama3.2  ✓
 2     qwen2.5-coder
 3     gpt-4o-mini

Enter number (0 to skip): 2
✓ MODEL_NAME saved as 'qwen2.5-coder'
```

### Method 2 — Inside chat: `/model`

Start any chat session and type `/model`:

```
phoenix ❯ /model
Fetching models...

 #     Model
 1     llama3.2  ✓
 2     qwen2.5-coder
 3     gpt-4o-mini

Enter number (0 to skip): 2
✓ switched to: qwen2.5-coder
```

Or skip the picker and switch directly:

```
phoenix ❯ /model llama3.2
✓ model switched to llama3.2
```

### Method 3 — `phoenix -m NAME "prompt"` (one-shot override)

```bash
phoenix -m gpt-4o -t 0.2 "Summarize this: $(cat notes.txt)"
```

This doesn't change your saved default — just this one request.

### Checking your current model

```bash
phoenix status
phoenix models          # current model is marked with ✓
```

---

## 4. MCP setup — Roblox MCP on mobile

**MCP (Model Context Protocol)** lets your AI call real tools — create Roblox
parts, edit scripts, query the workspace, etc. — directly from the chat.

### Step 1 — Enable MCP in your config

```bash
phoenix setup
# Answer "y" to: Enable MCP (Model Context Protocol) tools?
```

Or toggle it later:

```bash
# Just edit ~/.phoenix_config.json and set "mcp_enabled": true
```

### Step 2 — Install Node.js (Termux)

The Roblox MCP server runs on Node.js.

```bash
pkg install nodejs
```

### Step 3 — Add your Roblox MCP server

The easiest way is the one-command preset:

```bash
$ phoenix mcp add-roblox
```

It writes the correct command to `~/.phoenix_mcp.json` (and enables MCP if
needed):

```json
{
  "servers": [
    {
      "name": "roblox",
      "command": ["npx", "-y", "robloxstudio-mcp@latest"]
    }
  ]
}
```

> ⚠️ **Note:** older releases of this README used
> `npx -y @anthropic/mcp-server-roblox` — **that package does not exist on
> npm**, which is exactly why the server never connected. The preset uses the
> real, actively maintained `robloxstudio-mcp` package instead (a drop-in
> replacement with 50+ Studio tools). You can also add it manually with
> `phoenix mcp add` (transport `stdio`, command
> `npx -y robloxstudio-mcp@latest`).

`robloxstudio-mcp` bridges to **Roblox Studio through a local plugin**:

1. install the plugin from the
   [`robloxstudio-mcp` README](https://github.com/drgost1/robloxstudio-mcp)
   into Studio;
2. keep Roblox Studio open while chatting — the plugin shows
   **"Connected"** when the bridge is ready.

No API keys are required. The first `phoenix mcp test` may take a moment
while npx downloads the package.

### Step 4 — Test the connection

```bash
$ phoenix mcp test

Testing roblox... ✓ connected — 51 tool(s)
    🔧 create_object — Create a new instance
    🔧 get_file_tree — Browse the instance hierarchy as a tree
    🔧 edit_script — Edit a script in Studio
    🔧 set_property — Set any property on an instance
    ... and 47 more
```

If a server can't connect, Phoenix now prints the server's actual error
output (e.g. a missing npm package or a crashed process) instead of hanging
silently — `phoenix mcp test` is the fastest way to diagnose it.

### Step 5 — Chat with MCP tools enabled

```bash
$ phoenix

Model: gpt-4o-mini
API:   http://myserver.example.com:8080/v1
MCP:   12 tool(s) from 1 server(s)

phoenix ❯ create a red 4x4x1 brick called "Floor" in the workspace
  🔧 Calling mcp__roblox__create_part...
  ✓ Part "Floor" created at (0, 0, 0) with size (4, 4, 1), color red.
```

The AI calls the MCP tool **automatically** — no permission prompts. Toggle
this behavior in chat with `/auto on` (default) or `/auto off` (ask first).

### Managing MCP servers

```bash
phoenix mcp list            # see all configured servers
phoenix mcp test            # test every server
phoenix mcp test roblox     # test one server
phoenix mcp remove roblox   # remove a server
phoenix mcp add             # add another (supports SSE remote servers too)
```

### Adding a remote (SSE) MCP server

```bash
$ phoenix mcp add

Transport type [stdio]: sse
Server name: remote-tools
Server URL: https://my-mcp-server.example.com
API key for MCP server: sk-...

✓ MCP server 'remote-tools' added
```

---

## 5. All 30 chat commands

Start a chat session with `phoenix` (or `phoenix chat`), then use these commands. Every
command starts with `/`. Type `/help` at any time to see them all.

| # | Command | What it does |
|---|---------|-------------|
| 1 | `/help` | Show this complete command reference inside chat |
| 2 | `/exit`, `/quit`, `/q` | Leave the chat session (Ctrl+D also works) |
| 3 | `/clear` | Forget the conversation history (system prompt stays) |
| 4 | `/model` | **Interactive model picker** — fetches models and lets you choose by number |
| 5 | `/model NAME` | Switch directly to model `NAME` for this session |
| 6 | `/system` | Show the current system prompt |
| 7 | `/system TEXT` | Set a new system prompt (empty text clears it) |
| 8 | `/temp` | Show current sampling temperature |
| 9 | `/temp N` | Set temperature to `N` (e.g. `/temp 0.2` for focused, `/temp 0.9` for creative) |
| 10 | `/max-tokens` | Show current max-token cap |
| 11 | `/max-tokens N` | Cap reply length to `N` tokens |
| 12 | `/history` | Show how many messages are in memory and the trim limit |
| 13 | `/save` | Save the conversation to `phoenix_chat.md` |
| 14 | `/save FILE.md` | Save to a specific file (supports `/sdcard/...` paths) |
| 15 | `/tools` | List all MCP tools currently connected |
| 16 | `/mcp` | Show MCP server status, connected tools count, and auto-approve state |
| 17 | `/auto` | Show whether tool calls are auto-approved (default: ON) |
| 18 | `/auto on` | **Auto-approve tool calls** — the AI runs tools immediately, no prompts (default) |
| 19 | `/auto off` | Ask for confirmation before each tool call — review args before execution |
| 20 | `/status` | Show your current session config (model, URL, API key, MCP) |
| 21 | `/ping` | Send a tiny request to measure latency to your provider |
| 22 | `/models` | List the provider's available models inline (with numbered picker) |
| 23 | `/copy` | Copy the last AI reply to clipboard (Termux:API) or print it plain |
| 24 | `/undo` | Remove the last user+assistant exchange from history |
| 25 | `/retry` | Remove the last AI reply and resend your last message |
| 26 | `/export` | Export the full conversation as JSON to `phoenix_chat.json` |
| 27 | `/export FILE.json` | Export to a specific file |
| 28 | `/import FILE.json` | Import a conversation from a previously exported JSON file |
| 29 | `/theme NAME` | Switch code-block theme (`monokai`, `github_dark`, `dracula`, `vs_dark`) |
| 30 | `/verbose` | Toggle verbose mode (shows raw token chunks and tool payloads) |
| 31 | `/context` | Show current token count and estimated context window usage |
| 32 | `/reset` | Full reset — clear history, restore defaults, reload config |
| 33 | `/search WORD` | Search your conversation history (case-insensitive) and show matches |
| 34 | `/pin TEXT` | Pin a persistent note into the system prompt (survives /clear) |
| 35 | `/pinned` | List all pinned notes with numbers |
| 36 | `/unpin N` | Remove pinned note #N |
| 37 | `/compact` | Ask the AI to summarize the conversation into a compressed form (saves context) |
| 38 | `/config` | Show and interactively edit BASE_URL, API_KEY, MODEL_NAME |

### Key commands explained

**`/model`** — The interactive model picker. When you type `/model` with no
argument, Phoenix fetches all available models from your provider and shows a
numbered list. Type the number to switch. This is the easiest way to try
different models without leaving the chat.

**`/auto on` / `/auto off`** — Controls tool-call permission. When **ON**
(default), the AI calls MCP tools immediately — it can build Roblox games,
run code, query APIs, all without asking. When **OFF**, every tool call gets a
confirmation prompt so you can review the arguments first.

**`/pin TEXT`** — Pinned notes get added to the system prompt and survive
`/clear`. Great for rules like "always respond in Python" or "we're building a
Roblox obby game".

**`/compact`** — When conversations get long, use `/compact` to ask the AI to
summarize everything so far into a short message. The history gets replaced
with the summary, freeing up context window for new content.

**`/search WORD`** — Finds every message in your history containing `WORD`.
Shows the message number, role, and a snippet.

**`/copy`** — Copies the last assistant reply. On Termux, uses `termux-clipboard-set`.
On other systems, prints the reply in plain text (no ANSI codes) so you can
copy manually.

---

## 6. Single-prompt mode

For scripts, pipes, and quick questions:

```bash
phoenix "Write a python script that prints prime numbers up to 100"
phoenix "Explain git rebase in two sentences" --model llama3.2
phoenix -m gpt-4o -t 0.2 "Summarize this: $(cat notes.txt)"
phoenix --system "You are a terse Linux expert" "how do I mount an ext4 image?"
phoenix --no-stream "generate a JSON config for nginx" > nginx_idea.json
```

Options:

| Option            | Meaning                                        |
|-------------------|------------------------------------------------|
| `-m, --model`     | model override for this request                |
| `-s, --system`    | system prompt                                  |
| `-t, --temperature` | sampling temperature (float)                 |
| `--max-tokens N`  | cap the reply length                           |
| `--no-stream`     | print only the finished reply (nice for scripts/pipes) |

---

## 7. Provider cheat sheet

| Provider                | BASE_URL                              | API key needed? |
|-------------------------|---------------------------------------|-----------------|
| Ollama (local)          | `http://localhost:11434`              | no              |
| LM Studio (local)       | `http://localhost:1234`               | no              |
| vLLM (local)            | `http://localhost:8000`               | depends         |
| llama.cpp server        | `http://localhost:8080`               | depends         |
| LocalAI                 | `http://localhost:8080`               | no              |
| OpenRouter              | `https://openrouter.ai/api`           | yes             |
| Together AI             | `https://api.together.xyz`            | yes             |
| Groq                    | `https://api.groq.com/openai`         | yes             |
| DeepSeek                | `https://api.deepseek.com`            | yes             |
| Mistral                 | `https://api.mistral.ai`              | yes             |
| Fireworks               | `https://api.fireworks.ai/inference`  | yes             |
| xAI (Grok)              | `https://api.x.ai`                    | yes             |
| OpenAI                  | `https://api.openai.com`              | yes             |
| **Any custom server**   | `yourserver.example.com:port`         | depends         |

> **Tip:** You can type **any** base URL — Phoenix accepts bare hostnames,
> IPs with ports, custom subdomains, whatever. It will auto-add `http://` if
> you omit the scheme and `/v1` if there's no path.

---

## 8. Termux tips & troubleshooting

- **Ctrl key**: Termux puts `CTRL` on the extra-keys row above the keyboard
  (swipe it if hidden), or use volume-down button binding
  (*Termux: Settings → Volume keys*). Ctrl+C = cancel reply, Ctrl+D = exit chat.
- **Local servers on the same device**: if the model server runs *inside*
  Termux (`pkg install ollama`), bind it to `127.0.0.1` and use
  `http://localhost:11434`.
- **Local servers on another device** (e.g. Ollama on your PC): bind it to
  `0.0.0.0` (`OLLAMA_HOST=0.0.0.0:11434 ollama serve`), then use your PC's LAN
  IP: `phoenix setup` → `http://192.168.1.20:11434`. Verify with
  `phoenix status --probe`.
- **No network / DNS**: Termux networking works out of the box; if requests
  hang, toggle airplane mode or restart the app. `phoenix status --probe`
  tells you in seconds whether the problem is the URL or the connection.
- **Screen blanking mid-reply**: `termux-wake-lock` (Termux:API) keeps the CPU
  alive during long generations.
- **Terminal resizing**: rich re-measures the terminal on every render, so
  rotating the phone or splitting the pane reflows the output automatically.
- **Saving chats to shared storage**: `/save /sdcard/Documents/chat.md` needs
  `termux-setup-storage` first (one-time).
- **Installing extras fails to compile**: `[repl]` and `[highlight]` are pure
  Python, so they should never need a compiler.
- **Battery**: streaming is plain HTTPS; disable battery optimization for
  Termux only if you run long unattended generations.
- **Roblox MCP**: the server runs as a child process inside Termux. Make sure
  Node.js (`pkg install nodejs`) and the MCP package are installed. Check
  with `phoenix mcp test` before chatting.

---

## 9. Project layout & development

```
phoenix_cli/
├── __init__.py    version
── cli.py         click commands, chat loop, 30 slash commands, streaming UI
├── client.py      httpx streaming client, error taxonomy, tool-use support
├── config.py      ~/.phoenix_config.json handling, env-var overrides
└── mcp.py         MCP client (stdio + SSE), manager, tool definitions
tests/             pytest suite + a live mock OpenAI server + pty tests
assets/            logo and media for the README
pyproject.toml     packaging, `phoenix` entry point, extras
```

Tech stack: **Python 3.9+**, **httpx** (async streaming), **rich**
(terminal UI/markdown), **click** (CLI parsing) — with optional
**prompt_toolkit** and **pygments** extras.

### Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,repl,highlight]"
pytest                    # 60+ tests, incl. mock server + pty tests
```

### License

MIT
