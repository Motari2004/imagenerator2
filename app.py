import gradio as gr
from huggingface_hub import InferenceClient
import os

# --- Configuration ---
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Set environment variable HF_TOKEN.")

client = InferenceClient(token=HF_TOKEN, timeout=120)

# Professional CSS: Compact UI with fixed-size sharp image
custom_css = """
footer {visibility: hidden !important; display: none !important;}

.gradio-container {
    max-width: 500px !important; /* Smaller overall width for a compact feel */
    margin: 0 auto !important; 
    height: 98vh !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    gap: 5px !important;
}

#header-container {text-align: center; margin-bottom: 5px;}
#header-container h1 {font-weight: 900; font-size: 1.5rem; color: #ffffff; margin: 0;}

/* FIXED COMPACT IMAGE BOX */
#image-display {
    width: 350px !important; /* Forces the image to be smaller on screen */
    height: 350px !important;
    margin: 0 auto !important; /* Centers the image */
    aspect-ratio: 1 / 1 !important;
    background: #0f172a;
    border-radius: 15px;
    border: 1px solid rgba(255,255,255,0.1);
    object-fit: cover !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.5);
}

.input-row {
    background: rgba(255, 255, 255, 0.05);
    padding: 8px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.1);
}

.generate-btn {
    background: linear-gradient(90deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    min-width: 90px !important;
}

.status-text { font-size: 0.75rem; opacity: 0.7; }
"""

def generator(prompt):
    if not prompt or len(prompt.strip()) < 5:
        raise gr.Error("Prompt too short.")
    try:
        # Generate 1024 for high quality, but CSS will shrink it for display
        image = client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell",
            num_inference_steps=8,
            guidance_scale=0.0,
            height=1024,
            width=1024,
        )
        return image
    except Exception as e:
        raise gr.Error(f"System Error: {str(e)}")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=custom_css, title="Imagenerator Pro") as demo:
    
    with gr.Column(elem_id="header-container"):
        gr.Markdown("# ⚡ IMAGENERATOR")

    # Image Area: Constrained to 350px
    image_display = gr.Image(
        label=None,
        elem_id="image-display",
        type="pil",
        interactive=False,
        show_label=False,
        container=False
    )

    # Input Area: Inline and compact
    with gr.Row(elem_classes="input-row", equal_height=True):
        user_prompt = gr.Textbox(
            label=None,
            placeholder="Describe vision...",
            lines=1,
            container=False,
            scale=7,
            autofocus=True
        )
        generate_btn = gr.Button("Create", variant="primary", elem_classes="generate-btn", scale=2)

    # Minimal Status
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("Engine: **FLUX.1**", elem_classes="status-text")
        with gr.Column(scale=1):
            clear_btn = gr.Button("Reset", variant="link")

    # Events
    generate_btn.click(fn=generator, inputs=user_prompt, outputs=image_display)
    user_prompt.submit(fn=generator, inputs=user_prompt, outputs=image_display)
    clear_btn.click(lambda: (None, ""), outputs=[image_display, user_prompt])

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    try:
        demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)
    except OSError:
        demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)