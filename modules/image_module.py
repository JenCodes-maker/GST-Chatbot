"""
image_module.py
Predicts HSN code from a product image.
Uses BLIP (image captioning) to generate a description,
then passes it to TextGSTPredictor for HSN matching.
Falls back to EfficientNet label matching if BLIP unavailable.
"""

from modules.text_module import TextGSTPredictor


def get_image_description(image_path: str) -> str:
    """
    Generates a text description of the product image.
    Tries BLIP captioning first, falls back to basic label detection.
    """
    # ── Try BLIP image captioning ──────────────────────────────────────────
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        from PIL import Image
        import torch

        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )

        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt")

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=50)

        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption

    except Exception as e:
        print(f"[BLIP Error] {e} — using fallback")

    # ── Fallback: EfficientNet top-5 labels ───────────────────────────────
    try:
        import torch
        from torchvision import models, transforms
        from PIL import Image

        IMAGENET_LABELS_URL = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        import urllib.request
        with urllib.request.urlopen(IMAGENET_LABELS_URL) as f:
            labels = [line.strip().decode("utf-8") for line in f.readlines()]

        model = models.efficientnet_b0(pretrained=True)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]),
        ])

        img = Image.open(image_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)

        top5 = torch.topk(outputs, 5).indices[0].tolist()
        label_str = " ".join([labels[i].replace("_", " ") for i in top5])
        return label_str

    except Exception as e:
        print(f"[EfficientNet Error] {e} — using generic fallback")
        return "product goods item"


class ImageGSTPredictor:

    def __init__(self, dataset_path: str):
        self.text_predictor = TextGSTPredictor(dataset_path)

    def predict(self, image_path: str):
        """
        Generates image description → predicts HSN via text matching.
        Returns top 3 results + the detected description.
        """
        description = get_image_description(image_path)
        results = self.text_predictor.predict(description, top_n=3)
        return {
            "detected_description": description,
            "matches": results
        }
