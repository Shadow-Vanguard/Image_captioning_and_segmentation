import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
from transformers import pipeline
import numpy as np
import cv2

st.set_page_config(page_title="Caption + Segmentation", layout="wide")

@st.cache_resource
def load_captioner():
    return pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

@st.cache_resource
def load_semantic_model():
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT').eval()
    return model

@st.cache_resource
def load_instance_model():
    model = models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").eval()
    return model

def colorize_mask(mask):
    palette = [
        (0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128),
        (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0),
        (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128),
        (192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0),
        (0,64,128)
    ]
    h, w = mask.shape
    out = np.zeros((h,w,3), dtype=np.uint8)
    for cid in np.unique(mask):
        out[mask==cid] = palette[int(cid)%len(palette)]
    return Image.fromarray(out)

def overlay(image, mask_img, alpha=0.5):
    image = image.convert("RGBA")
    mask_img = mask_img.convert("RGBA")
    return Image.blend(image, mask_img, alpha)

def main():
    st.title("🖼️ Image Captioning + Segmentation")
    st.caption("Upload an image, get a caption, semantic segmentation, and instance segmentation.")

    left, right = st.columns([1,1])
    with left:
        upl = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
        example = st.checkbox("Use demo image", value=True if upl is None else False)
        if example:
            img = Image.open("demo_images/example.jpg").convert("RGB")
        else:
            if upl is None:
                st.stop()
            img = Image.open(upl).convert("RGB")
        st.image(img, caption="Input image", use_column_width=True)

    with right:
        if st.button("Run"):
            with st.spinner("Captioning..."):
                cap = load_captioner()
                text = cap(img)[0]["generated_text"]
            st.success("Caption: " + text)

            with st.spinner("Semantic segmentation..."):
                model = load_semantic_model()
                tfm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                x = tfm(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(x)["out"]
                mask = out.argmax(1).squeeze(0).cpu().numpy().astype("uint8")
                cm = colorize_mask(mask)
                ov = overlay(img, cm, 0.5)
                st.image(ov, caption="Semantic segmentation overlay", use_column_width=True)

            with st.spinner("Instance segmentation..."):
                model2 = load_instance_model()
                tfm2 = transforms.Compose([transforms.ToTensor()])
                x2 = tfm2(img)
                with torch.no_grad():
                    outputs = model2([x2])[0]
                img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                scores = outputs["scores"].cpu().numpy()
                boxes = outputs["boxes"].cpu().numpy()
                masks = outputs["masks"].cpu().numpy()  # [N,1,H,W]
                for i in range(len(scores)):
                    if scores[i] < 0.5: continue
                    box = boxes[i].astype(int)
                    mask = (masks[i,0] > 0.5).astype(np.uint8)
                    color = np.random.randint(0,255,3).tolist()
                    colored = np.zeros_like(img_bgr, dtype=np.uint8); colored[:] = color
                    img_bgr = np.where(mask[:,:,None]==1, img_bgr*0.5 + colored*0.5, img_bgr).astype(np.uint8)
                    cv2.rectangle(img_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Instance segmentation", use_column_width=True)

if __name__ == "__main__":
    main()
