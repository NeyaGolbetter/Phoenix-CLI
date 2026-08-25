# 🔥 Phoenix CLI

> **A provider-agnostic AI assistant for the terminal — built to run beautifully in Termux.**

Phoenix CLI is a lightweight, streaming chat client that talks the
**OpenAI-compatible API** format. Plug in *any* provider — **Ollama, LM Studio,
vLLM, llama.cpp, OpenRouter, Together AI, Groq, DeepSeek, Mistral, Fireworks,
xAI, LocalAI** — configure it once, and chat from anywhere, including your
Android phone.

```
$ phoenix "write a python script that prints prime numbers"
$ phoenix chat          # interactive conversation with history
$ phoenix setup         # configure provider
$ phoenix status        # check configuration
```

---

## Features

- **Custom provider support** — nothing is hardcoded. Any server that speaks
  the OpenAI Chat Completions protocol works: local (Ollama/LM Studio/vLLM)
  or cloud (OpenRouter/Together/Groq/…). You only set `BASE_URL`, `API_KEY`
  and `MODEL_NAME`.
- **Configuration management** — `phoenix setup` writes
  `~/.phoenix_config.json` (mode `0600`). Environment variables
  (`PHOENIX_BASE_URL`, `PHOENIX_API_KEY`, `PHOENIX_MODEL`) override the file,
  so secrets never have to touch disk.
- **Two interaction modes**
  - Single prompt: `phoenix "your question"`
  - Interactive chat: `phoenix chat` (conversation history kept in memory,
    auto-trimmed, with `/clear`, `/model`, `/save` and friends).
- **Token-by-token streaming** — replies render as they arrive, with a
  spinner while waiting for the first token.
- **Beautiful terminal UI via `rich`** — markdown rendering, syntax
  highlighting for code blocks (monokai), tables, panels, and colored output.
- **Termux-aware** — pure-Python dependency stack (no required C compilation,
  no GUI libraries), automatic handling of terminal resizes, graceful Ctrl+C
  that cancels the *request* instead of killing the app, and clean behavior
  when stdout is piped.
- **Precise error handling** — network failures, invalid API keys, unknown
  models, rate limits and wrong endpoints each produce a clear, actionable
  message (see [Error handling](#error-handling)).

---

## 1. Termux setup

Open Termux (from F-Droid) and run:

```bash
# 1. Update packages
pkg update && pkg upgrade

# 2. Install Python and git (Python 3.12 is fine; 3.9+ required)
pkg install python git

# 3. Get Phoenix CLI
git clone https://github.com/NeyaGolbetter/Phoenix-CLI
cd Phoenix-CLI

# 4. Install it (+ optional niceties, see below)
pip install .
pip install ".[repl,highlight]"     # optional: line editing + syntax colors
```

That's it. `phoenix` is now on your `$PATH` (`$PREFIX/bin`). If the shell
doesn't find it, start a new session or run `hash -r`, or use the fallback
`python -m phoenix_cli`.

> **Optional extras** (both pure Python — no compiler needed):
> `[repl]` = `prompt_toolkit` → command history, cursor keys and line editing
> in `phoenix chat`; `[highlight]` = `pygments` → full syntax highlighting in
> markdown code blocks. Without them, chat falls back to plain `input()` and
> code blocks render unhighlighted — everything else still works.

**Desktop (Linux/macOS/Windows WSL)?** Same steps, just skip the `pkg`
commands.

---

## 2. Configuration

### `phoenix setup`

Interactive, asks three questions (hidden input for the key):

```
$ phoenix setup

   ██████╗ ██╗  ██╗ ██████╗ ███████╗███╗   ██╗██╗██╗  ██╗
   ...
examples:
  Ollama (local)     http://localhost:11434
  LM Studio (local)  http://localhost:1234
  vLLM (local)       http://localhost:8000
  OpenRouter         https://openrouter.ai/api
  Together AI        https://api.together.xyz
  Groq               https://api.groq.com/openai
BASE_URL: http://localhost:11434
API_KEY (Enter for none — local servers usually need none):
MODEL_NAME: llama3.2

╭─ Saved ──────────────────────────────╮
│  BASE_URL   http://localhost:11434/v1│
│  API_KEY    (none)                   │
│  MODEL_NAME llama3.2                 │
╰──────────────────────────────────────╯
Config written to /data/data/com.termux/files/home/.phoenix_config.json
```

The config is a plain JSON file:

```json
{
  "base_url": "http://localhost:11434/v1",
  "api_key": "",
  "model_name": "llama3.2"
}
```

Notes:

- If `BASE_URL` has **no path**, `/v1` is appended automatically
  (`http://localhost:11434` → `http://localhost:11434/v1`). Explicit paths
  are kept as-is, so provider endpoints like `https://api.groq.com/openai`
  or `https://openrouter.ai/api` work unchanged.
- Environment variables take priority over the file:

  ```bash
  export PHOENIX_BASE_URL="http://192.168.1.20:11434"   # e.g. Ollama on your PC
  export PHOENIX_API_KEY="sk-..."
  export PHOENIX_MODEL="llama3.2"
  ```

- `PHOENIX_CONFIG=/path/to/file.json` points at an alternative config file
  (handy for multiple provider profiles).

### Provider cheat sheet

| Provider                | BASE_URL                     | API key needed? |
|-------------------------|------------------------------|-----------------|
| Ollama (local)          | `http://localhost:11434`     | no              |
| LM Studio (local)       | `http://localhost:1234`      | no              |
| vLLM (local)            | `http://localhost:8000`      | depends         |
| llama.cpp server        | `http://localhost:8080`      | depends         |
| LocalAI                 | `http://localhost:8080`      | no              |
| OpenRouter              | `https://openrouter.ai/api`  | yes             |
| Together AI             | `https://api.together.xyz`   | yes             |
| Groq                    | `https://api.groq.com/openai`| yes             |
| DeepSeek                | `https://api.deepseek.com`   | yes             |
| Mistral                 | `https://api.mistral.ai`     | yes             |
| Fireworks               | `https://api.fireworks.ai/inference` | yes      |
| xAI (Grok)              | `https://api.x.ai`           | yes             |
| OpenAI                  | `https://api.openai.com`     | yes             |

---

## 3. Usage

### Single prompt mode

```bash
phoenix "Write a python script that prints prime numbers up to 100"
phoenix "Explain git rebase in two sentences" --model llama3.2
phoenix -m gpt-4o -t 0.2 "Summarize this: $(cat notes.txt)"
phoenix --system "You are a terse Linux expert" "how do I mount an ext4 image?"
```

Options (work before or after the prompt):

| Option            | Meaning                                        |
|-------------------|------------------------------------------------|
| `-m, --model`     | model override for this request                |
| `-s, --system`    | system prompt                                  |
| `-t, --temperature` | sampling temperature (float)                 |
| `--max-tokens N`  | cap the reply length                           |
| `--no-stream`     | print only the finished reply (nice for scripts/pipes) |

Piping works cleanly — decorations are skipped and no ANSI codes leak:

```bash
phoenix --no-stream "generate a JSON config for nginx" > nginx_idea.json
```

### Interactive chat mode

```bash
phoenix chat
phoenix chat --model llama3.2 --system "You are my coding buddy"
```

```
Model: llama3.2
API:   http://localhost:11434/v1
Type /help for commands • Ctrl+C cancels a reply

phoenix ❯ what's a decorator in python?
Phoenix answers, streaming token by token...

phoenix ❯ /model qwen2.5-coder
✔ model switched to qwen2.5-coder
```

In-chat commands:

| Command            | Effect                                              |
|--------------------|-----------------------------------------------------|
| `/help`            | show all commands                                   |
| `/exit`, `/quit`   | leave the chat (Ctrl+D also works)                  |
| `/clear`           | forget conversation history                         |
| `/model NAME`      | switch model for this session                       |
| `/system TEXT`     | set/replace the system prompt (empty clears it)     |
| `/temp 0.8`        | set sampling temperature                            |
| `/max-tokens 1024` | cap reply length                                    |
| `/history`         | show how many messages are in memory                |
| `/save FILE`       | export the conversation as markdown                 |

History lives in memory (never on disk) and is automatically trimmed past 60
messages — oldest first, always keeping the system prompt and your latest
question — so long chats stay flat on a phone. Note that `phoenix
"prompt"` where the first word equals a command name (e.g. `phoenix
"status report"`) is treated as a command; use `phoenix ask "status report"`
for those cases.

### Other commands

```bash
phoenix status            # show config, where it comes from, masked key
phoenix status --probe    # + send a real test request and measure latency
phoenix --version
phoenix --help
```

---

## 4. Error handling

Every failure is caught and translated into a one-line diagnosis plus a hint.
Nothing crashes with a raw traceback.

| Error class             | When                                                              | Fix |
|-------------------------|-------------------------------------------------------------------|-----|
| `ConfigurationError`    | `BASE_URL` / `MODEL_NAME` missing or malformed                    | run `phoenix setup` |
| `APIKeyError`           | provider rejected the key (HTTP 401/403)                          | `phoenix setup` → new key |
| `ModelNotFoundError`    | model name unknown (HTTP 404, or a 400 mentioning the model)      | check the exact name (`ollama list`) |
| `RateLimitError`        | HTTP 429                                                          | wait, retry later |
| `NetworkError`          | DNS/refused/timeout/TLS — server unreachable                      | `phoenix status --probe`, check the URL |
| `ProviderError`         | HTTP 5xx or non-API answer (e.g. an HTML page → missing `/v1`)    | shown in the message |

```bash
$ phoenix "hi"
✖ APIKeyError
Authentication failed (HTTP 401). Your API key was rejected.
Run `phoenix setup` to update it.
```

**Ctrl+C behavior (Termux included):**

- mid-reply (single prompt or chat) → cancels the request, keeps the app alive,
  exit code `130` for the one-shot command;
- at the chat prompt → shows a hint instead of quitting (`/exit` or Ctrl+D);
- history stays consistent: a cancelled prompt is removed again, so the model
  never sees a question it didn't answer.

---

## 5. Termux tips & troubleshooting

- **Ctrl key**: Termux puts `CTRL` on the extra-keys row above the keyboard
  (swipe it if hidden), or use the volume-down button binding
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
- **Installing extras fails to compile**: `[repl]` and `[highlight]` are pure
  Python, so they should never need a compiler. If any dependency ever does,
  `pkg install rust binutils` is the Termux fix — but you don't need it here.
- **Saving chats to shared storage**: `/save /sdcard/Documents/chat.md` needs
  `termux-setup-storage` first (one-time).
- **Battery**: streaming is plain HTTPS; disable battery optimization for
  Termux only if you run long unattended generations.

---

## 6. Project layout

```
phoenix_cli/
├── __init__.py    version
├── cli.py         click commands, chat loop, streaming UI (rich.Live)
├── client.py      httpx streaming client, error taxonomy, conversation
└── config.py      ~/.phoenix_config.json handling, env-var overrides
tests/             pytest suite + a mock OpenAI-compatible server
pyproject.toml     packaging, `phoenix` entry point, extras
```

Tech stack: **Python 3.9+**, **httpx** (async streaming), **rich**
(terminal UI/markdown), **click** (CLI parsing) — with optional
**prompt_toolkit** and **pygments** extras.

## 7. Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,repl,highlight]"
pytest            # 42 tests, incl. a live mock OpenAI server + pty tests
```

## License

MIT
