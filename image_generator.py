import random
from io import BytesIO

import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image

st.set_page_config(
    page_title="SDXL Turbo Image Generator",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 SDXL Turbo Image Generator")

# ---------------- Load Model ---------------- #
@st.cache_resource
def load_model():
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    if torch.backends.mps.is_available():
        pipe.to("mps")
    else:
        pipe.to("cpu")

    pipe.set_progress_bar_config(disable=True)
    return pipe


pipe = load_model()

# ---------------- UI ---------------- #

prompt = st.text_area(
    "Prompt",
    placeholder="A futuristic cyberpunk city at sunset...",
    height=120,
)

negative_prompt = st.text_input(
    "Negative Prompt (Optional)",
    value="blurry, low quality, distorted"
)

col1, col2 = st.columns(2)

with col1:
    width = st.selectbox(
        "Width",
        [512, 768, 1024],
        index=0
    )

with col2:
    height = st.selectbox(
        "Height",
        [512, 768, 1024],
        index=0
    )

steps = st.slider(
    "Inference Steps",
    min_value=1,
    max_value=4,
    value=2,
)

seed = st.number_input(
    "Seed (0 = Random)",
    value=0,
    step=1,
)

generate = st.button(
    "Generate Image",
    use_container_width=True
)

# ---------------- Generation ---------------- #

if generate:

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    if seed == 0:
        seed = random.randint(1, 1_000_000)

    generator = torch.Generator(device="mps" if torch.backends.mps.is_available() else "cpu")
    generator.manual_seed(seed)

    with st.spinner("Generating image..."):

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

    st.success("Done!")

    st.image(image, caption=f"Seed: {seed}", use_container_width=True)

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    st.download_button(
        label="⬇ Download Image",
        data=buffer.getvalue(),
        file_name="generated_image.png",
        mime="image/png",
        use_container_width=True,
    )