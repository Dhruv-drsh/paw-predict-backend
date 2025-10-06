from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import io
import json
import time
import os

app = FastAPI(title="PawPredict API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
WEIGHTS_PATH = "models/efficientnet_b2_dog_classifier.h5"
LABELS_PATH = "models/dog_breeds_labels.json"

# Global variables
model = None
class_labels = []

def create_working_demo_model(num_classes):
    """Create a working model using ImageNet pre-trained weights"""
    print("🔧 Creating demo model with ImageNet features...")
    
    # Use pre-trained EfficientNet B2
    base_model = EfficientNetB2(
        weights='imagenet',  # Use ImageNet weights for feature extraction
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Keep base model frozen initially
    base_model.trainable = False
    
    # Add classification head
    inputs = base_model.input
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu', name='dense_2')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = Model(inputs, outputs)
    
    # Initialize the classification layers with reasonable weights
    # This helps give more realistic predictions
    for layer in model.layers[-6:]:  # Last few layers
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.assign(tf.random.normal(layer.kernel.shape, stddev=0.01))
        if hasattr(layer, 'bias_initializer'):
            layer.bias.assign(tf.zeros(layer.bias.shape))
    
    return model

@app.on_event("startup")
async def load_model():
    global model, class_labels
    try:
        # Load class labels
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        print(f"✅ Labels loaded: {len(class_labels)} breeds")
        
        # Check if original model exists and is valid
        file_size = os.path.getsize(WEIGHTS_PATH) if os.path.exists(WEIGHTS_PATH) else 0
        print(f"📏 Original model file size: {file_size} bytes")
        
        if file_size < 1000:  # Less than 1KB means empty/corrupted
            print("⚠️ Original model file is too small/corrupted")
            print("🔧 Creating demo model with ImageNet features...")
            
            # Create working demo model
            model = create_working_demo_model(len(class_labels))
            
            # Compile the model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✅ Demo model created and compiled successfully!")
            print("ℹ️ This model uses ImageNet features + random classification layer")
            print("ℹ️ Predictions will be based on general image features")
            
        else:
            print("🔄 Attempting to load your trained model...")
            # Try to load the trained model
            # (Keep your original loading logic here if file is large enough)
            
    except Exception as e:
        print(f"❌ Error in model loading: {e}")
        print("🔄 Falling back to basic model...")
        
        # Fallback: create basic model
        try:
            model = create_working_demo_model(len(class_labels))
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            print("✅ Fallback model created successfully!")
        except Exception as e2:
            print(f"❌ Critical error: {e2}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image for EfficientNet B2"""
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to array and preprocess for EfficientNet
    img_array = np.array(image, dtype=np.float32)
    
    # EfficientNet preprocessing (normalize to [-1, 1])
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_breed_info(breed_name: str) -> dict:
    """Enhanced breed information database"""
    breed_db = {
        "golden_retriever": {
            "temperament": "Golden Retrievers are friendly, intelligent, and devoted dogs. They make excellent family pets and are known for their patient nature with children.",
            "size": "Large (55-75 lbs)",
            "energy_level": "High",
            "good_with_kids": True,
            "life_span": "10-12 years",
            "origin": "Scotland"
        },
        "german_shepherd": {
            "temperament": "German Shepherds are confident, courageous, and smart working dogs. They are extremely versatile and loyal.",
            "size": "Large (50-90 lbs)",
            "energy_level": "High",
            "good_with_kids": True,
            "life_span": "9-13 years",
            "origin": "Germany"
        },
        "labrador_retriever": {
            "temperament": "Labs are friendly, outgoing, and active companions. They are among America's most popular dog breeds.",
            "size": "Large (55-80 lbs)",
            "energy_level": "High",
            "good_with_kids": True,
            "life_span": "10-12 years",
            "origin": "Newfoundland, Canada"
        },
        "bulldog": {
            "temperament": "Bulldogs are gentle, friendly, and calm. They make excellent companions and are good with children.",
            "size": "Medium (40-50 lbs)",
            "energy_level": "Low",
            "good_with_kids": True,
            "life_span": "8-10 years",
            "origin": "England"
        },
        "poodle": {
            "temperament": "Poodles are intelligent, active, and elegant. They are highly trainable and make great family pets.",
            "size": "Varies (toy, miniature, standard)",
            "energy_level": "Medium-High",
            "good_with_kids": True,
            "life_span": "12-15 years",
            "origin": "Germany/France"
        }
    }
    
    breed_key = breed_name.lower().replace(' ', '_').replace('-', '_')
    
    if breed_key in breed_db:
        return breed_db[breed_key]
    else:
        return {
            "temperament": f"{breed_name.replace('_', ' ').title()} dogs are known for their unique characteristics and make wonderful companions.",
            "size": "Medium",
            "energy_level": "Medium",
            "good_with_kids": True,
            "life_span": "10-14 years",
            "origin": "Various"
        }

@app.get("/")
async def root():
    return {"message": "🐕 PawPredict API is running!", "status": "demo_mode"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "total_breeds": len(class_labels),
        "mode": "demo_with_imagenet_features"
    }

@app.post("/predict")
async def predict_breed(file: UploadFile = File(...)):
    """Predict dog breed from uploaded image"""
    start_time = time.time()
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get breed name
        breed_name = class_labels[predicted_class]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "breed": class_labels[i].replace('_', ' ').title(),
                "confidence": float(predictions[0][i]),
                "percentage": round(float(predictions[0][i]) * 100, 2)
            }
            for i in top_3_indices
        ]
        
        # Get breed info
        breed_info = get_breed_info(breed_name)
        processing_time = f"{round(time.time() - start_time, 2)}s"
        
        return {
            "success": True,
            "prediction": {
                "breed": breed_name.replace('_', ' ').title(),
                "confidence": confidence,
                "percentage": round(confidence * 100, 2)
            },
            "top_predictions": top_3_predictions,
            "breed_info": breed_info,
            "image_size": list(image.size),
            "processing_time": processing_time,
            "note": "Demo mode: Using ImageNet features + random classification"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/breeds")
async def get_all_breeds():
    """Get list of all supported breeds"""
    return {
        "total_breeds": len(class_labels),
        "breeds": [breed.replace('_', ' ').title() for breed in class_labels]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
