import gradio as gr
from transformers import pipeline

# ✅ Use a summarization-trained model (light + good quality)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text):
    if not text.strip():
        return "⚠️ Please enter some text."
    # 🤏 Limit length for safety (avoid 512+ token OOM issues)
    truncated = text[:1500]
    result = summarizer(truncated, max_length=130, min_length=30, do_sample=False)
    return result[0]["summary_text"]

# Gradio UI
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=12, placeholder="Paste text here...", label="Input Text"),
    outputs=gr.Textbox(label="Summary"),
    title="📝 AI Text Summarizer",
    description="Summarize any text using Hugging Face Transformers `distilbart-cnn-12-6` model.",
)

if __name__ == "__main__":
    iface.launch()
