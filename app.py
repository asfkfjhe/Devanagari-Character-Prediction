import os
import numpy as np
import torch 

# import cv2
from fastapi import FastAPI, File, UploadFile
# from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse



app = FastAPI()

UPLOAD_DIR = "files"
MODEL_PATH = "model_0.pth"

# Load the saved model
model = torch.load(MODEL_PATH)
model.eval()  # Set the model to evaluation mode


# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Check if the file is an image (optional)
    if not file.content_type.startswith('image'):
        return {"message": "Only image files are allowed."}

    # Save the uploaded file to disk (optional)
    with open(f"{UPLOAD_DIR}/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    # Perform prediction
    prediction = int(predict_image(os.path.join(UPLOAD_DIR, file.filename)))

    return {"filename": file.filename, "prediction": prediction}

def predict_image(image_path):

    img = Image.open(image_path)
    img = transform(img)
    img = img.unsqueeze(0) 

    with torch.no_grad():
        output = model(img)
    
    # Assuming you want to get the class label with the highest probability
    predicted_class = torch.argmax(output).item()
    
    return predicted_class

