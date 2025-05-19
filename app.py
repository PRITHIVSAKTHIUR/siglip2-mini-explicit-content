import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/siglip2-mini-explicit-content"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated labels
labels = {
    "0": "Anime Picture",
    "1": "Extincing & Sensual",
    "2": "Hentai",
    "3": "Pornography",
    "4": "Safe for Work"
}

def detect_explicit_content(image):
    """Predicts content category in an uploaded image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    return predictions

# Gradio Interface
iface = gr.Interface(
    fn=detect_explicit_content,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="siglip2-mini-explicit-content",
    description="Upload an image to classify it as Anime, Hentai, Sensual, Pornographic, or Safe for Work."
)

if __name__ == "__main__":
    iface.launch()
