import io
import os
import numpy as np
from PIL import Image
try:
    import pillow_avif  # Allows processing .avif images
except ImportError:
    pass
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
import tensorflow as tf

app = FastAPI()

# Enable CORS for maximum compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "indian_food_model_v2.keras")
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "Indian Food Images")
IMG_SIZE = (300, 300)

model = None
class_labels = []

# Load labels dynamically
if os.path.exists(DATASET_DIR):
    class_labels = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
else:
    class_labels = ["biryani", "butter_chicken", "chaat", "chola_bhatura", "dosa", "gulab_jamun", "halwa", "idli", "naan", "pakora", "pav_bhaji", "rasgulla", "samosa", "tandoori_chicken"]

# Load model
if os.path.exists(MODEL_PATH):
    print("--- Loading Model... ---")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("--- Model Loaded Successfully! ---")
else:
    print(f"--- ERROR: Model not found at {MODEL_PATH} ---")

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Swad AI Dashboard</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; background: #0f0f0f; color: white; display: flex; flex-direction: column; align-items: center; padding: 40px; }
            .card { background: #181818; padding: 30px; border-radius: 20px; width: 100%; max-width: 450px; border: 1px solid #333; box-shadow: 0 10px 30px rgba(0,0,0,0.5); text-align: center; }
            h1 { color: #ff5722; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 30px; }
            input[type="file"] { background: #222; padding: 15px; width: 100%; border-radius: 10px; margin-bottom: 20px; box-sizing: border-box; border: 1px solid #444; color: #888; cursor: pointer; }
            button { background: linear-gradient(135deg, #ff5722, #ff7043); color: white; border: none; padding: 15px; border-radius: 12px; cursor: pointer; font-weight: bold; width: 100%; font-size: 1.1em; transition: 0.2s; }
            button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(255,87,34,0.3); }
            #result { margin-top: 30px; padding: 20px; background: #222; border-radius: 15px; border: 1px solid #333; display: none; }
            .prediction { font-size: 2em; font-weight: 800; color: #ff5722; margin: 10px 0; text-transform: uppercase; }
            .confidence { color: #4caf50; font-size: 1.1em; font-weight: 600; }
        </style>
    </head>
    <body>
        <h1>Swad AI Dashboard</h1>
        <div class="card">
            <input type="file" id="imageInput" accept="image/*">
            <button id="btn" onclick="predict()">ANALYZE IMAGE</button>
            <div id="result">
                <div id="details"></div>
            </div>
        </div>
        <script>
            async function predict() {
                const fileInput = document.getElementById('imageInput');
                const resultDiv = document.getElementById('result');
                const details = document.getElementById('details');
                const btn = document.getElementById('btn');
                if (!fileInput.files[0]) return alert("Please select a food image!");

                resultDiv.style.display = 'block';
                details.innerHTML = "Processing...";
                btn.disabled = true;

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();
                    details.innerHTML = `
                        <div style="font-size:0.8em; color:#666;">DISH DETECTED:</div>
                        <div class="prediction">${data.class_name}</div>
                        <div class="confidence">CONFIDENCE: ${(data.confidence * 100).toFixed(2)}%</div>
                    `;
                } catch (e) {
                    details.innerHTML = "<p style='color:#ff5252'>Error: " + e.message + "</p>";
                } finally {
                    btn.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None: raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_batch, verbose=0)
        idx = int(np.argmax(predictions[0]))
        name = class_labels[idx] if idx < len(class_labels) else f"Class {idx}"
        return {"class_name": name.replace("_", " "), "confidence": float(np.max(predictions[0]))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
