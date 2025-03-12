from fastapi import FastAPI, File, UploadFile
from uvicorn import run
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


origins = [
    "http://localhost:3000",  # React frontend
    "http://localhost",  # FastAPI backend (if needed)
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------

# Read the file as an image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  # Read the image and convert it to a numpy array
    return image

# Load the model with version 1
MODEL = tf.keras.models.load_model("src/models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Print model summary to verify it's loaded correctly
print(MODEL.summary())

# ----------------------------------------------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------------------------------------------

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Potato Disease Classification API!"}

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = read_file_as_image(await file.read())
        print("Image shape:", image.shape)
        
        image_batch = np.expand_dims(image, axis=0)
        print("Image batch shape:", image_batch.shape)
        
        predictions = MODEL.predict(image_batch)
        print("Predictions:", predictions)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions[0])
        
        return {
            "class": predicted_class,  
            "confidence": float(confidence)
        }
    except Exception as e:
        print(f"Error processing the file: {e}")
        return {"error": "Failed to process the image. Please ensure the file is a valid image."}

    
if __name__ == "__main__":
    run(app, host="localhost", port=8000)
