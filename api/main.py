from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the pre-trained model (choose the best one)
model = load_model('models/weights/best_model.h5')

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))  # Resize to match model input
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize
    return image

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes))
    image = np.array(image)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    result = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
    
    return {"prediction": result.tolist()}
