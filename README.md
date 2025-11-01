"# Image Captioning and Segmentation Project" 

his project combines **YOLOv8** for object detection and segmentation with **BLIP (Bootstrapped Language-Image Pre-training)** for image captioning.  
Together, they allow an AI system to **understand an image at pixel level** and **describe it in natural language**, bridging the gap between **computer vision** and **language understanding**.


Clone the Repository

git clone https://github.com/Shadow-Vanguard/Image_captioning_and_segmentation.git
cd Image_captioning_and_segmentation


Create Virtual Environment

python -m venv venv
venv\Scripts\activate


Install Dependencies

pip install -r requirements.txt

if shows error in installing requirements

python -m pip install --upgrade pip setuptools wheel



pip install numpy
pip install opencv-python
pip install matplotlib
pip install pillow
pip install tensorflow
pip install keras
pip install pandas
pip install scikit-learn
pip install nltk
pip install transformers
pip install tqdm
pip install jupyter




Run the App

python main.py


or for FastAPI:

uvicorn fastapi_server:app --reload



since my dataset was so large i didnt push it to my git