
![3.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/GbdtFwysvOM4Nulmetrtq.png)

# **siglip2-mini-explicit-content**

> **siglip2-mini-explicit-content** is an image classification vision-language encoder model fine-tuned from **`siglip2-base-patch16-512`** for a single-label classification task. It is designed to classify images into categories related to explicit, sensual, or safe-for-work content using the **SiglipForImageClassification** architecture.

> \[!Note]
> This model is intended to promote positive, safe, and respectful digital environments. Misuse is strongly discouraged and may violate platform or regional guidelines. As a classification model, it does not generate unsafe content and is suitable for moderation purposes.

> [!note]
*SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features* https://arxiv.org/pdf/2502.14786

> [!Important]
Note: Explicit, sensual, and pornographic content may appear in the results; however, all of them are considered not safe for work.

```py
Classification Report:
                     precision    recall  f1-score   support

      Anime Picture     0.8897    0.8296    0.8586      5600
Extincing & Sensual     0.8984    0.9477    0.9224      5618
             Hentai     0.8993    0.9118    0.9055      5600
        Pornography     0.9527    0.9285    0.9404      5970
      Safe for Work     0.8957    0.9172    0.9063      6000

           accuracy                         0.9074     28788
          macro avg     0.9072    0.9069    0.9066     28788
       weighted avg     0.9076    0.9074    0.9071     28788
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/PUJJPvJ4716zFSzluElry.png)

---

The model categorizes images into five classes:

* **Class 0:** Anime Picture
* **Class 1:** Extincing & Sensual
* **Class 2:** Hentai
* **Class 3:** Pornography
* **Class 4:** Safe for Work

---

# **Run with Transformers**

```python
!pip install -q transformers torch pillow gradio
```

```python
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
```

---

# **Guidelines for Use of siglip2-mini-explicit-content**

This model is designed for responsible content moderation and filtering. It is especially tuned for anime, hentai, and adult content. Use it ethically, with the following guidelines:

### **Recommended Use Cases**

* Content Moderation in social media and forums
* Parental Controls for safer browsing environments
* Dataset Curation for removing NSFW images from training data
* Safe Search Filtering for engines and discovery systems
* Workplace Image Scanning for compliance

### **Prohibited or Discouraged Use**

* Harassment, exposure, or targeting of individuals
* Use on private content without consent
* Illegal or unethical surveillance
* Sole reliance for legal or reputational decisions
* Deceptive manipulation of moderation results

---

# **Important Notes**

* Optimized for **anime and adult content detection**. Not suitable for detecting violence, drugs, or hate symbols.
* Probabilistic outputs — always **verify** with human review where needed.
* This model's predictions are **not legal classifications**.

---

## Demo Inference

> [!warning]
Anime Picture

![Screenshot 2025-05-19 at 21-47-04 siglip2-mini-explicit-content.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/Dl6Ltk-fXDaJR2ragkwa9.png)

> [!warning]
Extincing & Sensual

![Screenshot 2025-05-19 at 21-57-52 siglip2-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/w7i0P33ERa0bf9_05GBlF.png)
![Screenshot 2025-05-19 at 22-00-30 siglip2-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/PMJdqZrYyWaVDXdRqf7cq.png)

> [!warning]
Hentai

![Screenshot 2025-05-19 at 21-48-48 siglip2-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/lZFy9P94LoqxNFlVXZ8Jn.png)

> [!warning]
Pornography

![Screenshot 2025-05-19 at 22-07-04 siglip2-mini-explicit-content(1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/C79hg1c4g3rVAB5Wr8t2Z.png)

> [!warning]
Safe for Work

![Screenshot 2025-05-19 at 22-01-45 siglip2-mini-explicit-content.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/fltj5l5_SPxIzjC289C3U.png)

---

## **Ethical Reminder**

This tool was created to **enhance digital safety**. Do not use it to harm, surveil, or exploit individuals or communities. By using this model, you commit to ethical and privacy-respecting practices.
