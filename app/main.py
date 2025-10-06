from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
from datetime import datetime

app = FastAPI(title="Dog Breed Predictor API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = "models/best_phaseB.keras"
BREED_INFO_PATH = "models/breed_info.json"
CLASS_INDICES_PATH = "models/class_indices.json"

# Global variables
model = None
breed_database = {}
class_names = []

def load_breed_database():
    """Load breed information from JSON file"""
    global breed_database
    try:
        with open(BREED_INFO_PATH, 'r', encoding='utf-8') as f:
            breed_database = json.load(f)
        print(f"✓ Loaded {len(breed_database)} breeds from database")
        return True
    except FileNotFoundError:
        print(f"✗ Warning: {BREED_INFO_PATH} not found")
        return False
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing breed_info.json: {e}")
        return False

def load_class_indices():
    """Load class indices mapping"""
    global class_names
    try:
        with open(CLASS_INDICES_PATH, 'r', encoding='utf-8') as f:
            class_data = json.load(f)
        
        # Handle different formats
        if isinstance(class_data, dict):
            # If it's {0: "breed1", 1: "breed2"} format
            class_names = [class_data[str(i)] for i in range(len(class_data))]
        elif isinstance(class_data, list):
            # If it's already a list
            class_names = class_data
        
        print(f"✓ Loaded {len(class_names)} class names")
        return True
    except FileNotFoundError:
        print(f"✗ Warning: {CLASS_INDICES_PATH} not found")
        # Fallback to breed database keys
        class_names = list(breed_database.keys())
        return False
    except Exception as e:
        print(f"✗ Error loading class indices: {e}")
        class_names = list(breed_database.keys())
        return False

def load_model():
    """Load the trained model"""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"✗ Model file not found: {MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def normalize_breed_name(name):
    """Normalize breed name for consistent lookup"""
    return name.replace('_', ' ').replace('-', ' ').strip()

def get_breed_info(breed_name):
    """Get breed information from database"""
    normalized = normalize_breed_name(breed_name)
    
    # Try exact match (case-insensitive)
    for key in breed_database.keys():
        if normalize_breed_name(key).lower() == normalized.lower():
            return breed_database[key]
    
    # Return default if not found
    return {
        "size": "Medium",
        "temperament": ["Friendly", "Intelligent"],
        "energy_level": "Moderate",
        "life_span": "10-15 years",
        "group": "Not specified",
        "good_with_kids": "Unknown",
        "good_with_pets": "Unknown",
        "trainability": "Moderate",
        "origin": "Unknown",
        "exercise_needs": "Moderate",
        "grooming_needs": "Moderate",
        "barking_tendency": "Moderate",
        "bred_for": "Companionship",
        "weight_range": "Unknown",
        "height_range": "Unknown",
        "coat_type": "Unknown",
        "colors": ["Various"],
        "mental_stimulation_needs": "Moderate",
        "prey_drive": "Moderate",
        "sensitivity_level": "Moderate",
        "daily_food_amount": "Unknown",
        "calorie_requirements": "Unknown"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 50)
    print("Dog Breed Predictor API - Starting")
    print("=" * 50)
    
    load_breed_database()
    load_class_indices()
    
    model_loaded = load_model()
    
    if not model_loaded:
        print("\n⚠ WARNING: Model not loaded. API will not work properly.")
        print(f"Please ensure model file exists at: {MODEL_PATH}\n")
    
    print("=" * 50)
    print("API is ready!")
    print("=" * 50)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dog Breed Predictor API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "breeds_in_database": len(breed_database),
        "total_classes": len(class_names),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict dog breed from uploaded image"""
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is not ready."
        )
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG, WebP)"
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        processed_image = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_idx])
        
        # Get breed name
        if predicted_idx < len(class_names):
            breed_name = class_names[predicted_idx]
        else:
            breed_name = f"Unknown_Breed_{predicted_idx}"
        
        # Normalize breed name for display
        breed_display = normalize_breed_name(breed_name).title()
        
        # Get breed information
        breed_info = get_breed_info(breed_name)
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        
        for idx in top_3_indices:
            if idx < len(class_names):
                top_breed = normalize_breed_name(class_names[idx]).title()
                top_predictions.append({
                    "breed": top_breed,
                    "confidence": float(predictions[0][idx]),
                    "percentage": round(float(predictions[0][idx]) * 100, 2)
                })
        
        return {
            "success": True,
            "prediction": {
                "breed": breed_display,
                "confidence": confidence,
                "percentage": round(confidence * 100, 2)
            },
            "top_predictions": top_predictions,
            "breed_info": breed_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/breeds")
async def get_breeds():
    """Get list of all supported breeds"""
    breeds_list = [normalize_breed_name(b).title() for b in class_names]
    return {
        "total": len(breeds_list),
        "breeds": sorted(breeds_list)
    }

@app.get("/breed/{breed_name}")
async def get_breed_details(breed_name: str):
    """Get detailed information about a specific breed"""
    breed_info = get_breed_info(breed_name)
    
    if breed_info.get("size") != "Medium":  # Not default
        return {
            "breed": normalize_breed_name(breed_name).title(),
            "info": breed_info
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No information available for {breed_name}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)