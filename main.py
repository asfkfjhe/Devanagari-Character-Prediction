import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from fastapi.responses import JSONResponse



app = FastAPI()

UPLOAD_DIR = "uploaded_files"
MODEL_PATH = "model.h5"

# Load the saved model
model = load_model(MODEL_PATH)

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)




@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Check if the file is an image (optional)
    if not file.content_type.startswith('image'):
        return JSONResponse(status_code=415, content={"message": "Only image files are allowed."})

    # Save the uploaded file to disk (optional)
    with open(f"{UPLOAD_DIR}/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    # Perform prediction
    prediction = int(predict_image(os.path.join(UPLOAD_DIR, file.filename)))

    return {"filename": file.filename, "prediction": prediction}

def predict_image(image_path):

    # Load the trained model
    model = load_model('model.h5')
    
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # Normalize pixel values
    
    # Reshape the image to match the input shape expected by the model
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img)
    
    # Assuming you want to get the class label with the highest probability
    predicted_class = np.argmax(predictions)
    
    return predicted_class
