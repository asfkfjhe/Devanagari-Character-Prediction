from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import numpy as np
import cv2
import uvicorn
from torchvision import transforms
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
from classification_without_customds import TinyVGG  
import torch.nn as nn

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'aa', 'ae', 'ah', 'ai', 'an', 'ana', 'au', 'ba', 'bha',
        'cha', 'chha', 'da', 'daa', 'dha', 'dhaa', 'ee', 'ga', 'gha', 'gya', 'ha', 'i', 'ja', 'jha', 'ka', 'kha', 'kna',
        'ksha', 'la', 'ma', 'motosaw', 'na', 'o', 'oo', 'pa', 'patalosaw', 'petchiryosaw', 'pha', 'ra', 'ta', 'taa', 'tha', 'thaa', 'tra', 'u', 'va', 'ya', 'yna']
class TinyVGG(nn.Module):

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x
       


app = FastAPI()
model_path = "model_0.pth" 

try:
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model
    model.eval() 
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

preprocess = transforms.Compose([
    transforms.Resize((64, 64)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,)) 
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Only images allowed"})
    
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert('RGB') 
    
    input_tensor = preprocess(image).unsqueeze(0) 
    
    try:
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class_index = torch.argmax(output, dim=1).item() 
            predicted_class = labels[predicted_class_index] 

        return {"predicted_class": predicted_class}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Error predicting: {str(e)}"})


# Run the FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8004)
