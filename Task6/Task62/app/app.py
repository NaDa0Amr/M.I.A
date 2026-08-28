import os
import sys
import streamlit as st
from PIL import Image
import torch

# Add project root to path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.pipeline import CaptionPipeline
from src.utils.config import load_config

st.set_page_config(page_title='🖼️ Neural Image Captioning', layout='wide')

@st.cache_resource
def load_pipeline():
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return CaptionPipeline.from_checkpoint('checkpoints', device=device)
    except Exception as e:
        return str(e)

def main():
    st.sidebar.title("🖼️ Setup")
    strategy = st.sidebar.radio("Decoding Strategy", ["Beam Search", "Greedy"])
    
    beam_size = 3
    if strategy == "Beam Search":
        beam_size = st.sidebar.slider("Beam Size", 1, 10, 3)
        
    max_len = st.sidebar.slider("Max Caption Length", 5, 50, 20)
    
    st.sidebar.divider()
    st.sidebar.info("Neural Image Captioning model using ResNet-50 and LSTM.")
    
    st.title("🖼️ Neural Image Captioning")
    st.subheader("Upload an image to generate a description")
    
    pipeline_or_error = load_pipeline()
    
    if isinstance(pipeline_or_error, str):
        st.warning("Model checkpoints not found or failed to load. Please ensure you have trained the model.")
        st.error(f"Details: {pipeline_or_error}")
        st.info("Instructions: Run training script first and ensure 'checkpoints' directory exists in the project root.")
        return
        
    pipeline = pipeline_or_error
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png', 'webp'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
    with col2:
        if uploaded_file is not None:
            if st.button("Generate Caption", type="primary"):
                with st.spinner("Generating..."):
                    strat_val = 'beam' if strategy == 'Beam Search' else 'greedy'
                    try:
                        result = pipeline.generate(
                            image,
                            strategy=strat_val,
                            beam_size=beam_size,
                            max_length=max_len
                        )
                        st.success(result['caption'])
                        st.metric("Confidence Score", f"{result['score']:.4f}")
                    except Exception as e:
                        st.error(f"Generation error: {str(e)}")

if __name__ == "__main__":
    main()
