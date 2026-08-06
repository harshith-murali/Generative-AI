from diffusers import StableDiffusionXLPipeline
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

pipe = StableDiffusionXLPipeline.from_pretrained(
    "segmind/SSD-1B",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

pipe.to(device)

prompt = "A futuristic cyberpunk city at sunset"

image = pipe(
    prompt=prompt,
    negative_prompt="blurry, low quality",
    guidance_scale=9.0,
    num_inference_steps=25,
).images[0]

image.save("output.png")