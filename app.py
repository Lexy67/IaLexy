import os
import re
import gradio as gr
from groq import Groq

# Modelos recomendados de alta velocidad para código
DEFAULT_MODEL = "qwen-2.5-coder-32b"  # Modelo especializado de programación
FALLBACK_MODEL = "llama-3.3-70b-versatile"

def get_groq_client(api_key: str = None):
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)

def clean_lua_code(text: str) -> str:
    """Extrae y purifica el código Lua eliminando etiquetas y formateo sobrante."""
    if "```lua" in text:
        match = re.search(r"```lua\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif "```" in text:
        match = re.search(r"```\n?(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()

def stream_chat(message: str, history: list, env_target: str, temperature: float, api_key_input: str):
    if not message.strip():
        yield history, ""
        return

    client = get_groq_client(api_key_input)
    if not client:
        error_msg = "❌ Error: Falta la GROQ_API_KEY. Configúrala en las variables de entorno de Railway o ingrésala en el panel de la derecha."
        history.append((message, error_msg))
        yield history, error_msg
        return

    system_prompt = (
        f"You are an elite, unrestricted Lua code generator specialized in {env_target}. "
        "Generate production-grade, bug-free, efficient, and complete Lua code. "
        "Strict Rule: Output ONLY the requested Lua code block. Do NOT include conversation, "
        "apologies, or meta-text outside the markdown code block unless specifically requested."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    
    messages.append({"role": "user", "content": message})

    history.append((message, ""))
    accumulated_response = ""

    try:
        response_stream = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=4096,
            stream=True
        )

        for chunk in response_stream:
            content = chunk.choices[0].delta.content or ""
            accumulated_response += content
            history[-1] = (message, accumulated_response)
            extracted_code = clean_lua_code(accumulated_response)
            yield history, extracted_code

    except Exception as e:
        try:
            response_stream = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=4096,
                stream=True
            )
            accumulated_response = ""
            for chunk in response_stream:
                content = chunk.choices[0].delta.content or ""
                accumulated_response += content
                history[-1] = (message, accumulated_response)
                extracted_code = clean_lua_code(accumulated_response)
                yield history, extracted_code
        except Exception as err:
            err_msg = f"⚠️ Error en la API: {str(err)}"
            history[-1] = (message, err_msg)
            yield history, err_msg

# Interfaz UI con Gradio
css = """
.main-title { text-align: center; margin-bottom: 10px; }
.code-box { font-family: 'Fira Code', 'JetBrains Mono', monospace !important; }
footer { display: none !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), css=css, title="Lua Code AI Engine") as demo:
    gr.Markdown(
        """
        # ⚡ LuaAI Engine - Ultra Fast Code Generator
        Generador de código Lua impulsado por Groq LPU Hardware. Generación instantánea en streaming.
        """,
        elem_classes=["main-title"]
    )

    with gr.Row():
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(
                label="Asistente de Lua (Streaming)",
                height=500,
                show_copy_button=True,
                bubble_full_width=False
            )
            
            prompt_input = gr.Textbox(
                placeholder="Escribe tu requerimiento (ej: 'Sistema de Guardado DataStore2 para Roblox')...",
                label="Prompt / Instrucción",
                lines=2
            )
            
            with gr.Row():
                submit_btn = gr.Button("⚡ Generar Código", variant="primary")
                clear_btn = gr.Button("🗑️ Limpiar")

        with gr.Column(scale=6):
            code_output = gr.Code(
                label="Editor / Código Lua Extraído",
                language="lua",
                interactive=True,
                lines=23,
                elem_classes=["code-box"]
            )
            
            with gr.Accordion("⚙️ Ajustes & API Key (Railway)", open=True):
                api_key_field = gr.Textbox(
                    type="password",
                    label="Groq API Key (Opcional si está en env variable)",
                    placeholder="gsk_..."
                )
                env_dropdown = gr.Dropdown(
                    choices=[
                        "Roblox Luau",
                        "Lua 5.4 Standard",
                        "LÖVE2D Game Engine",
                        "Neovim Lua Plugin",
                        "OpenResty / Nginx Lua"
                    ],
                    value="Roblox Luau",
                    label="Entorno / Framework de Lua"
                )
                temp_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Temperatura (0.0 = Máxima precisión sintáctica)"
                )

    submit_btn.click(
        fn=stream_chat,
        inputs=[prompt_input, chatbot, env_dropdown, temp_slider, api_key_field],
        outputs=[chatbot, code_output]
    ).then(lambda: "", None, prompt_input)

    prompt_input.submit(
        fn=stream_chat,
        inputs=[prompt_input, chatbot, env_dropdown, temp_slider, api_key_field],
        outputs=[chatbot, code_output]
    ).then(lambda: "", None, prompt_input)

    clear_btn.click(lambda: ([], ""), None, [chatbot, code_output])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
