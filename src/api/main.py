from fastapi import FastAPI, File, UploadFile
from uvicorn import run
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI()

# ----------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------
def read_file_as_image(data) -> np.ndarray:
    image = np.Array(Image.open(BytesIO(data)))
    return image
# ----------------------------------------------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------------------------------------------


@app.get("/")
async def root():
    return {"Welcome to the Potato Disease Classification API!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    # Use wait and async to read the file asynchronously, when don't use async, the file will be read synchronously
    # If 100 users upload files at the same time, the server will read the files one by one. If using async, the server will read the files at the same time
    images = read_file_as_image(await file.read())

    return

if __name__ == "__main__":
    run(app, host="localhost", port=8000)
