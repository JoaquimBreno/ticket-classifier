import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import gradio as gr
import config
from src.rag import VectorStore
from src.graph import build_pipeline, run_pipeline

store = VectorStore.load(config.ARTIFACTS_DIR)
classes = sorted(set(store.labels))
compiled, _, _ = build_pipeline(store)


def classify(text: str) -> tuple[str, str]:
    if not (text and text.strip()):
        return "", "Digite o texto do ticket."
    try:
        out = run_pipeline(compiled, text.strip(), classes)
        return out.get("classe", ""), out.get("justificativa", "")
    except Exception as e:
        return "", f"Erro: {e}"


css = """
.gradio-container {
    background: #1a1a1a !important;
    display: flex !important;
    justify-content: center !important;
}
.gradio-container > div {
    max-width: 640px !important;
    width: 100% !important;
}
footer { display: none !important; }
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="slate", neutral_hue="slate")) as demo:
    gr.Markdown("## Classificador de tickets")
    inp = gr.Textbox(
        label="Ticket",
        placeholder="Cole o texto do ticket...",
        lines=6,
        max_lines=12,
    )
    btn = gr.Button("Classificar")
    with gr.Row():
        out_classe = gr.Textbox(label="Classe", interactive=False)
    out_just = gr.Textbox(label="Justificativa", interactive=False, lines=4)

    btn.click(fn=classify, inputs=inp, outputs=[out_classe, out_just])

if __name__ == "__main__":
    demo.launch()
