import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="centered",
)

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

st.title("🎨 AI Image Generator")
st.write("Generate stunning AI images using FLUX.1-dev")

prompt = st.text_area(
    "Enter your prompt",
    placeholder="A futuristic cyberpunk city at sunset, ultra realistic, 8k",
    height=120,
)

col1, col2 = st.columns(2)

with col1:
    width = st.selectbox("Width", [512, 768, 1024], index=2)

with col2:
    height = st.selectbox("Height", [512, 768, 1024], index=2)

if st.button("✨ Generate Image", use_container_width=True):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    with st.spinner("Generating image..."):

        image = client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-dev",
            width=width,
            height=height,
        )

    st.success("Done!")

    st.image(image, use_container_width=True)

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    st.download_button(
        label="⬇ Download Image",
        data=buffer.getvalue(),
        file_name="generated_image.png",
        mime="image/png",
        use_container_width=True,
    )