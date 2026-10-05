import os
import re
import tempfile
import time
import uuid

import gradio as gr
from groq import Groq

# ──────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────
ENVIRONMENTS = {
    "Roblox Luau": (
        "Roblox Luau. Use Roblox services (game:GetService), type annotations where useful, "
        "task.wait/task.spawn instead of wait/spawn, and Roblox best practices "
        "(RemoteEvents, ModuleScripts, Instance API)."
    ),
    "Lua 5.4": (
        "Standard Lua 5.4. Use only the standard library (string, table, math, io, os, utf8, coroutine). "
        "Use integer division, goto, <const>/<close> attributes where appropriate."
    ),
    "LÖVE2D": (
        "LÖVE2D (Lua 5.1/LuaJIT). Use the love.* callbacks (love.load, love.update, love.draw, "
        "love.keypressed, etc.) and the love.graphics / love.audio / love.physics modules."
    ),
    "Neovim (Lua)": (
        "Neovim Lua API. Use vim.api, vim.keymap.set, vim.opt, vim.fn, vim.cmd, autocommands "
        "via vim.api.nvim_create_autocmd, and lazy.nvim-style plugin specs when relevant."
    ),
    "LuaJIT / OpenResty": (
        "LuaJIT 2.1 with OpenResty/ngx_lua conventions where relevant. Use the FFI when useful."
    ),
    "Garry's Mod (GLua)": (
        "Garry's Mod GLua. Use hook.Add, net library, derma, and the standard GMod API."
    ),
}

BASE_SYSTEM_PROMPT = """You are an elite Lua code generator: an expert senior engineer who writes complete, \
clean, efficient and error-free Lua scripts. You answer every legitimate programming request directly and \
completely, without unnecessary warnings, hedging, moralizing or refusals.

TARGET ENVIRONMENT: {env_description}

STRICT OUTPUT RULES:
1. Respond with ONE single fenced code block tagged ```lua containing the full, ready-to-run script.
2. NO greetings, NO introductions, NO apologies, NO explanations, NO closing remarks outside the code block.
3. Any explanation must be written as concise Lua comments INSIDE the code.
4. Never use placeholders such as "-- TODO", "-- your code here" or "..." — write the entire implementation.
5. Write idiomatic, well-structured, performant code: local variables, small functions, clear names, \
proper error handling (pcall/error) when appropriate.
6. When the user asks to modify previous code, return the complete updated script, not a diff.
7. Only use APIs that exist in the target environment."""


def build_system_prompt(env_name: str) -> str:
    return BASE_SYSTEM_PROMPT.format(env_description=ENVIRONMENTS.get(env_name, ENVIRONMENTS["Lua 5.4"]))


# ──────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# Modelos preferidos (en orden). Se usan solo si tu cuenta tiene acceso a ellos.
PREFERRED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "qwen/qwen3-32b",
    "moonshotai/kimi-k2-instruct-0905",
    "llama-3.3-70b-versatile",
]
NON_CHAT_HINTS = ("whisper", "tts", "orpheus", "guard", "safeguard", "compound", "allam")


def discover_models():
    """Lista los modelos de chat a los que tu API key SÍ tiene acceso."""
    client = get_client()
    if client is not None:
        try:
            ids = [m.id for m in client.models.list().data]
            ids = [i for i in ids if not any(h in i.lower() for h in NON_CHAT_HINTS)]
            if ids:
                ordered = [m for m in PREFERRED_MODELS if m in ids]
                ordered += sorted(i for i in ids if i not in ordered)
                return ordered
        except Exception as e:  # noqa: BLE001
            print(f"[warn] No se pudo listar modelos de Groq: {e}")
    return list(PREFERRED_MODELS)


MODEL_CHOICES = discover_models()
DEFAULT_MODEL = os.getenv("GROQ_MODEL") or MODEL_CHOICES[0]
if DEFAULT_MODEL not in MODEL_CHOICES:
    MODEL_CHOICES.insert(0, DEFAULT_MODEL)


def strip_think(text: str) -> str:
    """Elimina bloques <think>...</think> (modelos de razonamiento como Qwen3)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)  # bloque aún abierto (streaming)
    return text.strip()


def extract_lua(text: str, final: bool = False) -> str:
    """Devuelve exclusivamente el código Lua puro, sin saludos ni explicaciones."""
    text = strip_think(text)

    # 1) Bloque cerrado ```lua ... ```
    match = re.search(r"```(?:lua|luau)?[ \t]*\r?\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 2) Bloque aún abierto (streaming en curso)
    match = re.search(r"```(?:lua|luau)?[ \t]*\r?\n(.*)$", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 3) Sin fences: solo en la respuesta final devolvemos el texto tal cual
    return text if final else ""


# ──────────────────────────────────────────────
# Lógica de chat con streaming
# ──────────────────────────────────────────────
def stats_md(model, started, chunks, state="✅ Listo"):
    secs = max(time.time() - started, 0.001)
    return f"{state} · 🧠 `{model}` · ⏱️ {secs:.1f}s · ⚡ ~{chunks / secs:.0f} chunks/s"


def respond(message, history, env, temperature, model):
    history = list(history or [])
    message = (message or "").strip()

    if not message:
        yield history, gr.update(), "", gr.update()
        return

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    yield history, gr.update(), "", "⏳ Conectando con Groq…"

    client = get_client()
    if client is None:
        history[-1]["content"] = (
            "⚠️ Falta la variable de entorno `GROQ_API_KEY`. "
            "Configúrala en Railway (pestaña Variables) y vuelve a desplegar."
        )
        yield history, gr.update(), "", "⚠️ Sin API key"
        return

    # Contexto: últimos 12 mensajes previos para ahorrar tokens
    previous = history[:-1][-12:]
    messages = [{"role": "system", "content": build_system_prompt(env)}]
    messages += [{"role": m["role"], "content": m["content"]} for m in previous]

    chosen = model or DEFAULT_MODEL
    candidates = [chosen] + [m for m in MODEL_CHOICES if m != chosen]
    last_error = None

    for candidate in candidates[:4]:
        full, chunks, started = "", 0, time.time()
        try:
            stream = client.chat.completions.create(
                model=candidate,
                messages=messages,
                temperature=float(temperature),
                max_tokens=8192,
                top_p=0.95,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                full += delta
                chunks += 1
                history[-1]["content"] = strip_think(full) or "…"
                yield history, extract_lua(full), "", stats_md(candidate, started, chunks, "⚡ Generando")

            history[-1]["content"] = strip_think(full) or "(Respuesta vacía)"
            yield history, extract_lua(full, final=True), "", stats_md(candidate, started, chunks)
            return

        except Exception as e:  # noqa: BLE001
            last_error = e
            err = str(e)
            if "model_not_found" in err or "does not exist" in err or "decommissioned" in err:
                continue  # probamos el siguiente modelo automáticamente
            break

    err = str(last_error)
    if "rate_limit" in err or "429" in err:
        friendly = "⏱️ Límite de uso de Groq alcanzado. Espera unos segundos e inténtalo de nuevo."
    elif "401" in err or "invalid_api_key" in err:
        friendly = "🔑 La `GROQ_API_KEY` no es válida. Revísala en Railway → Variables."
    else:
        friendly = f"❌ Error al llamar a Groq: `{err}`"
    history[-1]["content"] = friendly
    yield history, gr.update(), "", "❌ Error"


QUICK_ACTIONS = {
    "optimize": "Optimiza este código para máximo rendimiento y legibilidad. Devuelve el script completo.",
    "fix": "Revisa este código, corrige cualquier bug, error de sintaxis o uso de API incorrecto. Devuelve el script completo corregido.",
    "comment": "Añade comentarios claros y concisos a este código sin cambiar su lógica. Devuelve el script completo.",
    "extend": "Amplía este código con manejo de errores robusto, validaciones y una estructura modular. Devuelve el script completo.",
}


def quick_action(kind):
    def _run(code, history, env, temperature, model):
        if not (code or "").strip():
            yield history, gr.update(), "", "⚠️ Primero genera código en el panel derecho"
            return
        prompt = f"{QUICK_ACTIONS[kind]}\n\n```lua\n{code}\n```"
        yield from respond(prompt, history, env, temperature, model)

    return _run


def make_file(code):
    if not (code or "").strip():
        return gr.update(value=None, interactive=False)
    folder = os.path.join(tempfile.gettempdir(), "lua_ai")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"script_{uuid.uuid4().hex[:8]}.lua")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return gr.update(value=path, interactive=True)


def clear_all():
    return [], "", "", "🧹 Conversación limpia", gr.update(value=None, interactive=False)


# ──────────────────────────────────────────────
# Interfaz
# ──────────────────────────────────────────────
CSS = """
.gradio-container {max-width: 1600px !important; margin: auto;}
#hero {text-align: center; padding: 18px 12px 6px;}
#hero h1 {font-size: 2.3rem; margin: 0; font-weight: 800;
  background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
#hero p {opacity: .75; margin: 6px 0 10px;}
.badge {display: inline-block; padding: 3px 12px; margin: 0 4px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; border: 1px solid rgba(128,128,128,.35);}
.badge.ok {background: rgba(34,197,94,.15); color: #22c55e;}
.badge.bad {background: rgba(239,68,68,.15); color: #ef4444;}
.panel {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 10px;}
#stats {text-align: center; opacity: .8; font-size: .85rem; min-height: 24px;}
footer {display: none !important;}
"""


def make_code_component():
    kwargs = dict(label="📄 Código Lua limpio", interactive=True, lines=26, max_lines=40)
    try:
        return gr.Code(language="lua", **kwargs)
    except Exception:  # versiones de Gradio sin resaltado Lua
        return gr.Code(language=None, **kwargs)


EXAMPLES = [
    "Sistema de inventario con ranuras, apilado de ítems y guardado/carga",
    "Controlador de personaje en tercera persona con sprint, salto doble y cámara suave",
    "Sistema de combate con cooldowns, daño, críticos y efectos de estado",
    "Tienda de monedas con DataStore, validación en servidor y RemoteEvents",
    "Juego Snake completo con puntuación y niveles",
    "Plugin de Neovim para alternar comentarios y navegar buffers",
    "Clase OOP con herencia, metatables y un sistema de eventos",
    "Algoritmo A* para pathfinding en una cuadrícula",
]

key_ok = bool(os.getenv("GROQ_API_KEY"))
theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

with gr.Blocks(theme=theme, css=CSS, title="Lua AI Generator") as demo:
    gr.HTML(
        "<div id='hero'><h1>🌙 Lua AI Generator</h1>"
        "<p>Scripts Lua completos y limpios en segundos, con la velocidad de Groq LPU</p>"
        f"<span class='badge {'ok' if key_ok else 'bad'}'>"
        f"{'● API conectada' if key_ok else '● Falta GROQ_API_KEY'}</span>"
        f"<span class='badge'>{len(MODEL_CHOICES)} modelos disponibles</span></div>"
    )

    with gr.Row():
        env_dd = gr.Dropdown(choices=list(ENVIRONMENTS.keys()), value="Roblox Luau",
                             label="🎯 Entorno objetivo", scale=2)
        model_dd = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="🧠 Modelo Groq",
                               allow_custom_value=True, scale=2)
        temp_sl = gr.Slider(0.0, 1.5, value=0.3, step=0.05, scale=2,
                            label="🌡️ Temperatura (bajo = preciso, alto = creativo)")

    stats = gr.Markdown("💤 Listo para generar", elem_id="stats")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes="panel"):
            chatbot = gr.Chatbot(type="messages", label="💬 Chat", height=500, show_copy_button=True)
            msg = gr.Textbox(placeholder="Describe el script de Lua que necesitas… (Enter para enviar)",
                             label="Tu petición", lines=2, max_lines=6)
            with gr.Row():
                send_btn = gr.Button("🚀 Generar", variant="primary", scale=3)
                stop_btn = gr.Button("⏹️ Detener", variant="stop", scale=1)
                clear_btn = gr.Button("🗑️ Limpiar", scale=1)
            gr.Examples(examples=EXAMPLES, inputs=msg, label="💡 Ejemplos (clic para usar)")

        with gr.Column(scale=1, elem_classes="panel"):
            code_out = make_code_component()
            with gr.Row():
                b_opt = gr.Button("⚡ Optimizar", size="sm")
                b_fix = gr.Button("🐞 Corregir", size="sm")
                b_com = gr.Button("📝 Comentar", size="sm")
                b_ext = gr.Button("🧱 Ampliar", size="sm")
            download_btn = gr.DownloadButton("⬇️ Descargar .lua", interactive=False, variant="secondary")

    base_in = [env_dd, temp_sl, model_dd]
    chat_in = [msg, chatbot] + base_in
    outs = [chatbot, code_out, msg, stats]
    quick_in = [code_out, chatbot] + base_in

    events = [
        msg.submit(respond, chat_in, outs),
        send_btn.click(respond, chat_in, outs),
        b_opt.click(quick_action("optimize"), quick_in, outs),
        b_fix.click(quick_action("fix"), quick_in, outs),
        b_com.click(quick_action("comment"), quick_in, outs),
        b_ext.click(quick_action("extend"), quick_in, outs),
    ]
    for ev in events:
        ev.then(make_file, code_out, download_btn)

    stop_btn.click(lambda: "⏹️ Detenido", None, stats, cancels=events)
    clear_btn.click(clear_all, None, [chatbot, code_out, msg, stats, download_btn])

demo.queue(default_concurrency_limit=10)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)), show_api=False)
