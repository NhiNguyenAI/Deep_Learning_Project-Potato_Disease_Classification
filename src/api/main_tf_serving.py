from fastapi import FastAPI, File, UploadFile
from uvicorn import run
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import requests

app = FastAPI()

# ----------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------

# Read the file as an image
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  # Read the image and convert it to a numpy array
    return image

# Load the model with version 1
# PROD_MODEL = tf.keras.models.load_model("src/models/1")
# BASE_MODEL = tf.keras.models.load_model("src/models/1")

endpoint = "http://localhost:8501/v1/models/potatoes:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ----------------------------------------------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------------------------------------------

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Potato Disease Classification API!"}

# Predict endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)  # Ensure the field name is "file"
):
    try:
        # Read the image file asynchronously
        image = read_file_as_image(await file.read())
        
        # Add a batch dimension (batch_size = 1)
        image_batch = np.expand_dims(image, axis=0)
        
        # Get predictions from the model
        json_data = {
            "instances": image_batch.tolist()
        }
        reponse = requests.post(endpoint, json = json_data)

        pass

    except Exception as e:
        # Log the error and return a meaningful message
        print(f"Error processing the file: {e}")
        return {"error": "Failed to process the image. Please ensure the file is a valid image."}

if __name__ == "__main__":
    run(app, host="localhost", port=8000)