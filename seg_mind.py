import io

import streamlit as st
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

st.set_page_config(page_title="SSD-1B Image Generator", page_icon="🎨")

st.title("🎨 SSD-1B + SDXL Lightning")

# -------------------------------------------------
# Device
# -------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

# -------------------------------------------------
# Load Model (cached)
# -------------------------------------------------
@st.cache_resource
def load_model():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "segmind/SSD-1B",
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        use_safetensors=True,
    )

    # Load Lightning LoRA
    pipe.load_lora_weights(
        "ByteDance/SDXL-Lightning",
        weight_name="sdxl_lightning_4step_lora.safetensors",
    )

    pipe.fuse_lora()

    # Lightning Scheduler
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
    )

    pipe.to(device)

    # Memory optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()

    return pipe


pipe = load_model()

# -------------------------------------------------
# UI
# -------------------------------------------------
prompt = st.text_area(
    "Prompt",
    "Ultra realistic futuristic cyberpunk city at sunset, neon lights, cinematic lighting, masterpiece, highly detailed, 8k",
)

negative_prompt = st.text_area(
    "Negative Prompt",
    "low quality, blurry, watermark, text, logo, deformed, duplicate, bad anatomy",
)

seed = st.number_input(
    "Seed",
    min_value=0,
    max_value=999999,
    value=42,
)

# -------------------------------------------------
# Generate
# -------------------------------------------------
if st.button("🎨 Generate Image", type="primary"):

    if not prompt.strip():
        st.warning("Please enter a prompt.")

    else:

        generator = torch.Generator(device=device).manual_seed(seed)

        with st.spinner("Generating image..."):

            with torch.inference_mode():

                image = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=4,      # Lightning is designed for 4 steps
                    guidance_scale=0.0,         # Recommended for Lightning
                    generator=generator,
                    width=768,
                    height=768,
                ).images[0]

        st.image(image, caption="Generated Image", use_container_width=True)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")

        st.download_button(
            "📥 Download PNG",
            data=buffer.getvalue(),
            file_name="generated_image.png",
            mime="image/png",
        )

