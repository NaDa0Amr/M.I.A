import os
import sys
import gradio as gr
import torch
from PIL import Image
import nltk

# Download NLTK data required for metrics/vocabulary on Hugging Face Spaces
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Add project root to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import CaptionPipeline

# Global pipeline instance
pipeline = None

def load_model():
    global pipeline
    if pipeline is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            pipeline = CaptionPipeline.from_checkpoint('checkpoints', device=device)
            return f"✅ Model loaded successfully on **{device}**."
        except Exception as e:
            return f"❌ Failed to load model: {e}\n\nMake sure you have trained the model and `checkpoints/` exists."
    return f"✅ Model ready."

def generate_caption(image, strategy, beam_size, max_length, temperature):
    if image is None:
        return "⚠️ Please upload an image first."
    if pipeline is None:
        return "❌ Model not loaded properly."

    try:
        # Generate caption using the pipeline
        result = pipeline.generate(
            image,
            strategy="beam" if strategy == "Beam Search" else "greedy",
            beam_size=int(beam_size),
            max_length=int(max_length),
            temperature=float(temperature)
        )
        print(f"Generated caption: {result['caption']}")
        return f"Caption: {result['caption']}\nConfidence Score: {result['score']:.4f}"
    except Exception as e:
        print(f"Error: {e}")
        return f"Error generating caption: {e}"

# Build Gradio Interface using Blocks
with gr.Blocks(title="Neural Image Captioning", theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🖼️ Neural Image Captioning")
    gr.Markdown("Upload an image to generate a description using a CNN-LSTM model trained on Flickr8k.")
    
    status_text = gr.Markdown(load_model())
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            
            with gr.Accordion("Advanced Settings", open=False):
                strategy_input = gr.Radio(["Beam Search", "Greedy"], value="Beam Search", label="Decoding Strategy")
                beam_size_input = gr.Slider(1, 10, value=3, step=1, label="Beam Size (for Beam Search)")
                max_len_input = gr.Slider(5, 50, value=20, step=1, label="Max Caption Length")
                temp_input = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature")
                
            submit_btn = gr.Button("Generate Caption", variant="primary")
            
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Generated Output", lines=4)
            
    submit_btn.click(
        fn=generate_caption,
        inputs=[image_input, strategy_input, beam_size_input, max_len_input, temp_input],
        outputs=output_text
    )

if __name__ == "__main__":
    # Run the Gradio app (defaults to port 7860, binds to 127.0.0.1 locally so the link works on Windows)
    interface.launch(server_port=7860, share=False)
