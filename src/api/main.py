from fastapi import FastAPI, File, UploadFile
from uvicorn import run
from PIL import Image
import io

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Potato Disease Classification API!"}

@app.get("/ping")
async def ping():
    return {"message": "Pong!"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    ):
    pass

if __name__ == "__main__":
    run(app, host="localhost", port=8000)
