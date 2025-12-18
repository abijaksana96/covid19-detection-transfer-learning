from http.client import HTTPException
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

model_path = "app/model/tf_learning_with_vgg16.h5" # untuk path model (ganti sesuai lokasi model Anda)

try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Nama kelas
class_name = np.array(['normal', 'covid'])

# Inisialisasi FastAPI
app = FastAPI()

# Menggunakan CORS Middleware untuk mengizinkan permintaan dari domain lain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Semua domain diizinkan
    allow_credentials = True, # Diperlukan untuk autentikasi
    allow_methods = ["*"], # Mengizinkan semua method
    allow_headers = ["*"]
)

def preprocess_image(image_bytes):
    """
    Preprocessing image sebelum diprediksi
    """
    try:
        # Load image
        image = Image.open(io.BytesIO(image_bytes))

        # Konversi gambar ke RGB (3 channel) jika saat ini 4 channel (RGBA)
        if image.mode == 'RGBA':
            image = image.convert('RGB')

        # Resize image
        image = image.resize((224, 224))

        # Konversi PIL Image ke NumPy array
        image_array = keras_image.img_to_array(image)

        # Menambahkan dimensi batch
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the COVID-19 Classification API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Baca gambar
        image_data = await file.read()
        
        # Preprocess gambar
        image_array = preprocess_image(image_data)

        # Buat prediksi
        predictions = model.predict(image_array)
        probabilities = float(predictions[0])

        # Konversi ke nama kelas
        predicted_class = class_name[1] if probabilities < 0.5 else class_name[0]
        confidence = 1 - probabilities if probabilities < 0.5 else probabilities

        return {
            "filename": file.filename,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": {
                "covid": round((1 - probabilities) * 100, 2),
                "normal": round(probabilities * 100, 2)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")
    
