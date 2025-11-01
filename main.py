import os
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from itsdangerous import URLSafeTimedSerializer
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import Optional

# Configuration
SECRET_KEY = os.environ.get("SECRET_KEY", "super_secret_key") # Use environment variable for production
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
USERS = {
    "example@gmail.com": "password123"
}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# FastAPI app setup
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load YOLO model (Segmentation)
yolo_model = YOLO("yolov8x-seg.pt") 

# Load BLIP (Captioning)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Utility functions
def generate_caption(image_path: str) -> str:
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def run_segmentation(image_path: str, output_path: str):
    results = yolo_model.predict(image_path, conf=0.15)
    for r in results:
        im_bgr = r.plot()
        im = Image.fromarray(im_bgr[..., ::-1])  
        im.save(output_path)
    detections = []
    for r in results:
        for c in r.boxes.cls:
            detections.append(yolo_model.names[int(c)])
    return detections

# Dependency to check if user is logged in
async def get_current_user(request: Request):
    if not request.session.get('logged_in'):
        raise HTTPException(status_code=status.HTTP_303_SEE_OTHER, detail="Not authenticated",
                            headers={"Location": "/login?error=You must be logged in to view this page"})
    return request.session.get('logged_in')

# Routes
@app.get("/login", response_class=HTMLResponse)
async def login_get(request: Request, error: Optional[str] = None):
    response = templates.TemplateResponse("login.html", {"request": request, "error": error})
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, email: str = Form(...), password: str = Form(...)):
    if email in USERS and USERS[email] == password:
        request.session['logged_in'] = True
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    else:
        response = templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"})
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

@app.get("/logout")
async def logout(request: Request):
    request.session.pop('logged_in', None)
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user_status: bool = Depends(get_current_user)):
    response = templates.TemplateResponse("index.html", {"request": request})
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile = File(...), user_status: bool = Depends(get_current_user)):
    if not file:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    with open(filepath, "wb") as buffer:
        buffer.write(await file.read())

    caption = generate_caption(filepath)

    output_filename = filename.split('.')[0] + "_segmented.jpeg"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    detections = run_segmentation(filepath, output_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_image": request.url_for('static', path=f'uploads/{filename}'),
        "segmented_image": request.url_for('static', path=f'outputs/{output_filename}'),
        "caption": caption,
        "detections": detections
    })
