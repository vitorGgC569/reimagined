"""
Web chat UI for NSOS v10_gpu.

Single-file Python HTTP server with embedded HTML/CSS/JS.  No external
deps (no Flask/FastAPI/etc).  Loads the v10_gpu model on GPU and serves
a modern chat interface at http://127.0.0.1:8765 — open it in a browser
and start talking.

Layout:
  ┌─ header (model info, GPU/CPU badge) ──────────┐
  │  sidebar     │     chat                       │
  │  - tasks     │     ┌──── user bubble ────┐    │
  │  - settings  │     └─────────────────────┘    │
  │              │     ┌──── ai bubble ──────┐    │
  │              │     └─────────────────────┘    │
  │              │   ┌── input + send ────────┐   │
  └──────────────────┴────────────────────────────┘
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# UTF-8 stdout
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

WORKTREE  = Path(__file__).resolve().parent.parent.parent.parent
BUILD_DIR = Path("C:/Users/Oxta/Desktop/reimagined-main/.claude/worktrees/"
                 "clever-roentgen-ba007c/OXN/nsos/build-cuda-validation")
RUN_DIR   = WORKTREE / "OXN/nsos/scripts/live_distill_v10_gpu"
MODEL_BIN = RUN_DIR / "phase6_memory.bin"
MODEL_CFG = RUN_DIR / "effective_model_config.json"

if str(BUILD_DIR) not in sys.path:
    sys.path.insert(0, str(BUILD_DIR))
if os.name == "nt":
    for p in [BUILD_DIR,
              Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")]:
        try:
            if p.exists():
                os.add_dll_directory(str(p))
        except (AttributeError, OSError) as exc:
            print(
                f"[runtime] DLL directory registration failed for {p}: {exc}",
                file=sys.stderr,
            )

import nsos_ext as nsos  # type: ignore  # noqa: E402

USE_GPU = os.environ.get("NSOS_CHAT_DEVICE", "gpu").lower() == "gpu"

print("Loading model...")
config = nsos.ModelConfig()
if MODEL_CFG.exists():
    for k, v in json.loads(MODEL_CFG.read_text("utf-8")).items():
        if hasattr(config, k):
            setattr(config, k, v)
config.use_cuda = USE_GPU

engine = nsos.InferenceEngine()
t0 = time.time()
if not engine.load_model(str(MODEL_BIN), config):
    print(f"[ERROR] load failed", file=sys.stderr)
    sys.exit(1)
LOAD_S = time.time() - t0
DEVICE_LABEL = "GPU (CUDA)" if USE_GPU else "CPU"
print(f"[ready] model loaded in {LOAD_S:.1f}s on {DEVICE_LABEL}")
print(f"[ready] vocab={config.vocab_size}  layers={config.num_layers}  d_model={config.d_model}")

# Serialize inference calls (engine is not thread-safe across requests)
GEN_LOCK = threading.Lock()


def build_prompt(kind: str, user_text: str) -> str:
    return f"<|task:{kind}|>\nPrompt:\n{user_text}\nAnswer:\n"


# ──────────────────────────────────────────────────────────────────────
# HTML page (single string).  Modern dark theme, glassmorphic accents.
# ──────────────────────────────────────────────────────────────────────
HTML_PAGE = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSOS v10 — chat</title>
<style>
  :root{
    --bg: #0b1020;
    --bg-2: #0f1730;
    --panel: rgba(255,255,255,0.05);
    --panel-strong: rgba(255,255,255,0.09);
    --border: rgba(255,255,255,0.10);
    --text: #e8ecf5;
    --muted: #8b94ad;
    --accent: #7aa2ff;
    --accent-2: #b388ff;
    --user: #1f3a8a;
    --ai: #1f2937;
    --ok: #34d399;
    --warn: #fbbf24;
    --err: #f87171;
  }
  *{box-sizing:border-box}
  html,body{margin:0;padding:0;height:100%;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:var(--text);background:var(--bg)}
  body{
    background:
      radial-gradient(1200px 800px at 10% -10%, rgba(122,162,255,0.18), transparent 60%),
      radial-gradient(800px 600px at 110% 10%, rgba(179,136,255,0.14), transparent 60%),
      linear-gradient(180deg, var(--bg) 0%, var(--bg-2) 100%);
    min-height: 100vh;
  }
  .app{display:grid;grid-template-columns:280px 1fr;grid-template-rows:60px 1fr;height:100vh;}
  /* header */
  header{
    grid-column: 1 / -1;
    display:flex;align-items:center;gap:12px;
    padding: 0 18px;
    border-bottom:1px solid var(--border);
    backdrop-filter: blur(8px);
    background: rgba(11,16,32,0.55);
    z-index:10;
  }
  .logo{
    width:32px;height:32px;border-radius:9px;
    background:linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow:0 4px 20px rgba(122,162,255,0.4);
    display:grid;place-items:center;font-weight:800;color:#0b1020;
  }
  header .title{font-weight:700;letter-spacing:0.2px}
  header .sub{color:var(--muted);font-size:12px;margin-left:6px}
  header .spacer{flex:1}
  .badge{
    padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;
    border:1px solid var(--border);background:var(--panel);
  }
  .badge.ok{color:var(--ok);border-color:rgba(52,211,153,0.35);background:rgba(52,211,153,0.08)}
  .badge.warn{color:var(--warn);border-color:rgba(251,191,36,0.35);background:rgba(251,191,36,0.08)}

  /* sidebar */
  aside{
    border-right:1px solid var(--border);
    background: rgba(15,23,48,0.4);
    padding:14px;
    overflow-y:auto;
  }
  .sec-title{font-size:11px;text-transform:uppercase;letter-spacing:1.4px;color:var(--muted);margin:14px 8px 6px}
  .task-btn{
    display:flex;align-items:center;gap:10px;
    width:100%;padding:10px 12px;margin-bottom:4px;
    border:1px solid var(--border);
    background:var(--panel);
    border-radius:10px;color:var(--text);font-size:14px;cursor:pointer;
    text-align:left;
    transition: all 120ms ease;
  }
  .task-btn:hover{background:var(--panel-strong);transform:translateX(2px)}
  .task-btn.active{
    border-color:var(--accent);
    background:linear-gradient(135deg, rgba(122,162,255,0.20), rgba(179,136,255,0.10));
    box-shadow:0 4px 14px rgba(122,162,255,0.18);
  }
  .task-btn .ico{
    width:24px;height:24px;border-radius:6px;background:var(--panel-strong);
    display:grid;place-items:center;font-size:13px;
  }
  .task-help{font-size:11px;color:var(--muted);padding:2px 8px 8px;line-height:1.4}

  .setting{display:flex;align-items:center;justify-content:space-between;padding:6px 8px;font-size:13px}
  .setting input[type=range]{width:130px}
  .setting input[type=number]{
    width:70px;background:var(--panel);border:1px solid var(--border);color:var(--text);
    border-radius:6px;padding:4px 6px;font-size:12px;
  }
  .setting .val{color:var(--muted);font-size:12px;min-width:40px;text-align:right}

  /* chat */
  main{display:flex;flex-direction:column;overflow:hidden}
  .chat{
    flex:1;overflow-y:auto;padding:24px 24px 12px;
    display:flex;flex-direction:column;gap:14px;
  }
  .msg{display:flex;gap:10px;max-width:780px}
  .msg.user{align-self:flex-end;flex-direction:row-reverse}
  .avatar{
    width:32px;height:32px;border-radius:50%;flex:none;
    display:grid;place-items:center;font-size:13px;font-weight:700;
    box-shadow:0 2px 8px rgba(0,0,0,0.3);
  }
  .msg.user .avatar{background:linear-gradient(135deg, #60a5fa, #3b82f6);color:white}
  .msg.ai .avatar{background:linear-gradient(135deg, var(--accent), var(--accent-2));color:#0b1020}
  .bubble{
    padding:12px 14px;border-radius:14px;line-height:1.5;font-size:14.5px;
    word-wrap:break-word;white-space:pre-wrap;
    border:1px solid var(--border);
  }
  .msg.user .bubble{background:linear-gradient(135deg, #2563eb, #4f46e5);color:white;border-color:transparent}
  .msg.ai .bubble{background:var(--ai)}
  .bubble.error{background:rgba(248,113,113,0.10);border-color:rgba(248,113,113,0.35);color:#fda4a4}
  .meta{font-size:11px;color:var(--muted);margin-top:4px}
  .msg .task-tag{
    font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
    color:var(--muted);padding:2px 7px;border:1px solid var(--border);border-radius:999px;display:inline-block;
    margin-bottom:6px;
  }

  /* input */
  .input-wrap{
    border-top:1px solid var(--border);
    padding:14px 24px 18px;
    background:rgba(11,16,32,0.55);
    backdrop-filter:blur(6px);
  }
  .input-row{
    display:flex;gap:10px;align-items:flex-end;
    background:var(--panel);border:1px solid var(--border);border-radius:14px;
    padding:10px;
  }
  .input-row textarea{
    flex:1;background:transparent;border:none;outline:none;color:var(--text);
    font-family:inherit;font-size:14.5px;resize:none;min-height:24px;max-height:140px;
    line-height:1.5;
  }
  .send-btn{
    background:linear-gradient(135deg, var(--accent), var(--accent-2));
    color:#0b1020;font-weight:700;border:none;border-radius:10px;
    padding:8px 18px;cursor:pointer;font-size:14px;
    box-shadow:0 4px 14px rgba(122,162,255,0.32);
    transition:transform 100ms ease, box-shadow 100ms ease;
  }
  .send-btn:hover{transform:translateY(-1px);box-shadow:0 6px 18px rgba(122,162,255,0.45)}
  .send-btn:disabled{opacity:0.5;cursor:not-allowed;transform:none;box-shadow:none}

  /* typing dots */
  .typing{display:inline-flex;gap:4px;align-items:center}
  .typing span{
    width:6px;height:6px;border-radius:50%;background:var(--muted);
    animation:bounce 1.2s infinite ease-in-out;
  }
  .typing span:nth-child(2){animation-delay:0.15s}
  .typing span:nth-child(3){animation-delay:0.30s}
  @keyframes bounce{
    0%,80%,100%{transform:translateY(0);opacity:0.4}
    40%{transform:translateY(-4px);opacity:1}
  }

  .warn-banner{
    margin: 0 24px 8px;padding:8px 12px;border-radius:10px;font-size:12px;
    background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.30);
    color:#fde68a;
  }

  /* scrollbar */
  ::-webkit-scrollbar{width:10px;height:10px}
  ::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.10);border-radius:8px}
  ::-webkit-scrollbar-thumb:hover{background:rgba(255,255,255,0.18)}

  @media (max-width: 720px){
    .app{grid-template-columns:1fr;grid-template-rows:60px auto 1fr}
    aside{max-height:200px;border-right:none;border-bottom:1px solid var(--border)}
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="logo">N</div>
    <div>
      <div class="title">NSOS v10_gpu <span class="sub">phase6_memory.bin</span></div>
    </div>
    <div class="spacer"></div>
    <span id="device-badge" class="badge ok">__DEVICE__</span>
    <span class="badge">vocab __VOCAB__ • d_model __DMODEL__ • L __LAYERS__</span>
    <span id="speed-badge" class="badge">— tok/s</span>
  </header>

  <aside>
    <div class="sec-title">Tasks</div>
    <button class="task-btn active" data-kind="summarize">
      <div class="ico">∑</div><div><div>summarize</div><div class="task-help">resumir texto</div></div>
    </button>
    <button class="task-btn" data-kind="extract_fact">
      <div class="ico">?</div><div><div>extract_fact</div><div class="task-help">contexto + pergunta</div></div>
    </button>
    <button class="task-btn" data-kind="explain_code">
      <div class="ico">{}</div><div><div>explain_code</div><div class="task-help">explicar snippet</div></div>
    </button>
    <button class="task-btn" data-kind="rewrite">
      <div class="ico">✎</div><div><div>rewrite</div><div class="task-help">reescrever texto</div></div>
    </button>
    <button class="task-btn" data-kind="translate">
      <div class="ico">⇄</div><div><div>translate</div><div class="task-help">traduzir</div></div>
    </button>

    <div class="sec-title">Settings</div>
    <div class="setting">
      <span>max_tokens</span>
      <input type="number" id="max_tokens" min="8" max="256" value="48" step="8">
    </div>
    <div class="setting">
      <span>temperature</span>
      <input type="range" id="temp" min="0" max="1.5" step="0.05" value="0.4">
      <span class="val" id="temp-val">0.40</span>
    </div>
    <div class="setting">
      <span>top_p</span>
      <input type="range" id="top_p" min="0.1" max="1.0" step="0.05" value="0.85">
      <span class="val" id="top_p-val">0.85</span>
    </div>
    <div class="setting">
      <span>top_k</span>
      <input type="number" id="top_k" min="1" max="200" value="20">
    </div>
  </aside>

  <main>
    <div class="warn-banner">
      ⚠️ Modelo viu apenas ~1.2M tokens (vs 750M-1.3B Chinchilla). Espere palavras
      inglesas reconhecíveis mas <b>sem coerência</b> — é proof-of-life arquitetural, não LLM.
    </div>
    <div id="chat" class="chat"></div>
    <div class="input-wrap">
      <div class="input-row">
        <textarea id="input" placeholder="Digite sua pergunta...  (Enter envia • Shift+Enter quebra linha)" rows="1"></textarea>
        <button id="send" class="send-btn">Enviar</button>
      </div>
    </div>
  </main>
</div>

<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const speedBadge = document.getElementById('speed-badge');
let currentKind = 'summarize';
let busy = false;

// task buttons
document.querySelectorAll('.task-btn').forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll('.task-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentKind = btn.dataset.kind;
    input.focus();
  };
});

// live setting labels
const tempEl = document.getElementById('temp');
const tempVal = document.getElementById('temp-val');
tempEl.oninput = () => tempVal.textContent = parseFloat(tempEl.value).toFixed(2);
const tpEl = document.getElementById('top_p');
const tpVal = document.getElementById('top_p-val');
tpEl.oninput = () => tpVal.textContent = parseFloat(tpEl.value).toFixed(2);

// auto-grow textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 140) + 'px';
});

function makeBubble(role, text, opts = {}) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'You' : 'AI';
  const box = document.createElement('div');
  const tag = opts.kind ? `<div class="task-tag">${opts.kind}</div>` : '';
  const meta = opts.meta ? `<div class="meta">${opts.meta}</div>` : '';
  box.innerHTML = tag +
    `<div class="bubble${opts.error ? ' error' : ''}">${text}</div>` + meta;
  div.appendChild(avatar);
  div.appendChild(box);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return box.querySelector('.bubble');
}

function makeTyping(kind) {
  const div = document.createElement('div');
  div.className = 'msg ai';
  div.innerHTML = `
    <div class="avatar">AI</div>
    <div>
      <div class="task-tag">${kind}</div>
      <div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>
    </div>`;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function escapeHtml(s){
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
}

async function send() {
  if (busy) return;
  const text = input.value.trim();
  if (!text) return;
  busy = true;
  sendBtn.disabled = true;

  makeBubble('user', escapeHtml(text));
  input.value = '';
  input.style.height = 'auto';

  const typingEl = makeTyping(currentKind);

  const payload = {
    kind: currentKind,
    prompt: text,
    max_tokens: parseInt(document.getElementById('max_tokens').value, 10) || 48,
    temperature: parseFloat(tempEl.value),
    top_p: parseFloat(tpEl.value),
    top_k: parseInt(document.getElementById('top_k').value, 10) || 20,
  };

  try {
    const t0 = performance.now();
    const res = await fetch('/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    typingEl.remove();

    if (!data.ok) {
      makeBubble('ai', escapeHtml(data.error || 'erro desconhecido'),
        {kind: currentKind, error: true});
    } else {
      const meta = `${data.tokens} tokens • ${data.elapsed_s.toFixed(1)}s • ${data.tok_s.toFixed(2)} tok/s`;
      makeBubble('ai', escapeHtml(data.text || '(resposta vazia)'),
        {kind: currentKind, meta});
      speedBadge.textContent = data.tok_s.toFixed(2) + ' tok/s';
    }
  } catch (e) {
    typingEl.remove();
    makeBubble('ai', 'falha na requisição: ' + escapeHtml(e.message),
      {kind: currentKind, error: true});
  } finally {
    busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

sendBtn.onclick = send;
input.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

// kick: greet bubble
makeBubble('ai',
  'Olá. Sou o NSOS v10_gpu (~40M params, ~1.2M tokens vistos). ' +
  'Escolha uma task à esquerda e digite uma pergunta. Aviso: não esperem coerência — gera palavras inglesas reais mas sem semântica forte.',
  {kind: 'system'});

input.focus();
</script>
</body>
</html>
"""

# Substituir placeholders (não usar .format por causa do CSS com {)
HTML_PAGE = (HTML_PAGE
             .replace("__DEVICE__", DEVICE_LABEL)
             .replace("__VOCAB__", str(config.vocab_size))
             .replace("__DMODEL__", str(config.d_model))
             .replace("__LAYERS__", str(config.num_layers)))


# ──────────────────────────────────────────────────────────────────────
# HTTP handler
# ──────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # quieter logs
        sys.stderr.write(f"[http] {self.address_string()} - {fmt % args}\n")

    def _send_json(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/healthz":
            self._send_json(200, {"ok": True, "device": DEVICE_LABEL})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            req = json.loads(raw)
        except Exception as e:
            self._send_json(400, {"ok": False, "error": f"bad request: {e}"})
            return

        kind = (req.get("kind") or "summarize").strip()
        prompt_text = (req.get("prompt") or "").strip()
        if not prompt_text:
            self._send_json(400, {"ok": False, "error": "prompt vazio"})
            return

        # Live tuning of generation options per request
        with GEN_LOCK:
            opts = nsos.GenerationOptions()
            opts.max_context_tokens = 512
            opts.max_tokens   = max(1, min(int(req.get("max_tokens", 48)), 256))
            opts.temperature  = float(req.get("temperature", 0.4))
            opts.top_p        = float(req.get("top_p", 0.85))
            opts.top_k        = max(1, int(req.get("top_k", 20)))
            opts.stream       = False

            full_prompt = build_prompt(kind, prompt_text)
            t = time.time()
            try:
                out = engine.generate_ex(full_prompt, opts)
            except Exception as e:
                self._send_json(500, {"ok": False, "error": f"generate failed: {e}"})
                return
            elapsed_s = time.time() - t
            out = out.replace("<|endoftext|>", "").strip()
            m = engine.last_generation_metrics()
            tok_s = m.generated_tokens / (m.elapsed_ms / 1000.0) if m.elapsed_ms > 0 else 0.0

        self._send_json(200, {
            "ok": True,
            "text": out,
            "tokens": m.generated_tokens,
            "elapsed_s": elapsed_s,
            "tok_s": tok_s,
        })


HOST = "127.0.0.1"
PORT = int(os.environ.get("NSOS_CHAT_PORT", "8765"))

print()
print(f"  ┌──────────────────────────────────────────┐")
print(f"  │  NSOS v10 chat UI is ready                │")
print(f"  │  → http://{HOST}:{PORT}/                  │")
print(f"  │  device: {DEVICE_LABEL:<28}    │")
print(f"  │  Ctrl+C to stop                           │")
print(f"  └──────────────────────────────────────────┘")
print()

try:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.serve_forever()
except KeyboardInterrupt:
    print("\n[server] shutting down")
    server.server_close()
