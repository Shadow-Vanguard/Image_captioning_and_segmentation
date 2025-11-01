import torch
from transformers import pipeline

class Captioner:
    def __init__(self):
        # device setup
        if torch.cuda.is_available():
            self.device = 0
        else:
            self.device = -1

        # BLIP captioning pipeline
        self.pipe = pipeline(
            task="image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=self.device
        )

    @torch.inference_mode()
    def generate(self, image_path: str) -> str:
        try:
            out = self.pipe(image_path, max_new_tokens=30)
            # pipeline returns a generated_text
            return out[0]["generated_text"].strip()
        except Exception as e:
            # safe fallback so app doesn't crash
            return f"(captioning fallback) Could not generate caption: {e}"
