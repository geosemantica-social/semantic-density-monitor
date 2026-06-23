import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
from sd_monitor import SDMonitor

# Load model and tokenizer
MODEL_NAME = "EleutherAI/pythia-410m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

monitor = SDMonitor(model, tokenizer)

# Default examples
DEFAULT_PAIRS = {
    "English": {
        "coherent": "The system operates within normal parameters.",
        "broken": "Parameters normal within operates system the.",
    },
    "Spanish": {
        "coherent": "La inteligencia artificial transforma nuestra comprensión del lenguaje.",
        "broken": "Subatómicas partículas de comportamiento el describe cuántica teoría la.",
    },
}


def analyze(coherent_text, broken_text):
    if not coherent_text or not broken_text:
        return {"Error": "Please provide both texts."}
    results = monitor.evaluate(coherent_text, broken_text)
    return results


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(
            label="Coherent Text",
            placeholder="Enter a grammatically correct sentence...",
            lines=3,
            value=DEFAULT_PAIRS["English"]["coherent"],
        ),
        gr.Textbox(
            label="Broken Syntax Text",
            placeholder="Enter the syntactically degraded counterpart...",
            lines=3,
            value=DEFAULT_PAIRS["English"]["broken"],
        ),
    ],
    outputs=gr.JSON(label="Semantic Density Metrics"),
    title="Semantic Density Phase Inversion Monitor",
    description=(
        "Computes ΔS = S(coherent) - S(broken). "
        "ΔS < 0 indicates the model is in a stable representational regime.\n\n"
        "Based on: The Stone Guest (Cerda Seguel, 2025)."
    ),
    examples=[
        [DEFAULT_PAIRS["English"]["coherent"], DEFAULT_PAIRS["English"]["broken"]],
        [DEFAULT_PAIRS["Spanish"]["coherent"], DEFAULT_PAIRS["Spanish"]["broken"]],
    ],
    cache_examples=True,
)

demo.launch()