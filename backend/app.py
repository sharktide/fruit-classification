from fastapi import FastAPI, File, UploadFile, Request
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import tensorflowtools as tft

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tft.hftools.download_model_from_huggingface('sharktide', 'fruitbot0', 'tf_model.keras')
tft.hftools.download_model_from_huggingface('sharktide', 'fruitbot1', 'tf_model.keras')

fruitbot0 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot0", "tf_model.keras")
fruitbot1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot1", "tf_model.keras")

FRUITBOT_CLASSES = ['Apple 10', 'Apple 11', 'Apple 12', 'Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19', 
           'Apple 5', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple Core 1', 'Apple Red Yellow 2', 'Apple worm 1', 
           'Banana 3', 'Beans 1', 'Blackberrie 1', 'Blackberrie 2', 'Blackberrie half rippen 1', 
           'Blackberrie not rippen 1', 'Cabbage red 1', 'Cactus fruit green 1', 'Cactus fruit red 1', 'Caju seed 1', 
           'Cherimoya 1', 'Cherry Wax not rippen 1', 'Cucumber 10', 'Cucumber 9', 'Gooseberry 1', 'Pistachio 1', 
           'Quince 2', 'Quince 3', 'Quince 4', 'Tomato 1', 'Tomato 5', 'apple_6', 'apple_braeburn_1', 
           'apple_crimson_snow_1', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith_1', 
           'apple_hit_1', 'apple_pink_lady_1', 'apple_red_1', 'apple_red_2', 'apple_red_3', 'apple_red_delicios_1', 
           'apple_red_yellow_1', 'apple_rotten_1', 'cabbage_white_1', 'carrot_1', 'cucumber_1', 'cucumber_3', 
           'eggplant_long_1', 'pear_1', 'pear_3', 'zucchini_1', 'zucchini_dark_1']

# Create FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preprocess the image (resize, reshape without normalization)
def preprocess_image(image_file, model):
    try:
        # Load image using PIL
        image = Image.open(image_file)
        
        # Convert image to numpy array
        image = np.array(image)

        if model == "fruitbot0": 
            image = cv2.resize(image, (240, 240))
            image = image.reshape(-1, 240, 240, 3)
        elif model == "fruitbot1":
            image = cv2.resize(image, (224, 224))
            image = image.reshape(-1, 224, 224, 3)
        
        return image
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

@app.get("/predict")
def predict():
    return JSONResponse(content={"Models Avalible For Inference at this Endpoint": ["fruitbot0", "fruitbot1"], "Models Avalible For Inference at Another Endpoint": ["recyclebot0"], "All Models": ["fruitbot0", "fruitbot1", "recyclebot0"]})

@app.post("/predict/fruitbot0")
async def predict_fruitbot0(file: UploadFile = File(...)):
    try:
        logger.info("Received request for /predict/fruitbot0")
        img_array = preprocess_image(file.file, "fruitbot0")  # Preprocess the image
        prediction1 = fruitbot0.predict(img_array)  # Get predictions
        
        predicted_class_idx = np.argmax(prediction1, axis=1)[0]  # Get predicted class index
        predicted_class = FRUITBOT_CLASSES[predicted_class_idx]  # Convert to class name
        
        return JSONResponse(content={"prediction": predicted_class})
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predict/fruitbot1")
async def predict_fruitbot0(file: UploadFile = File(...)):
    try:
        logger.info("Received request for /predict/fruitbot1")
        img_array = preprocess_image(file.file, "fruitbot1")  # Preprocess the image
        prediction1 = fruitbot1.predict(img_array)  # Get predictions
        
        predicted_class_idx = np.argmax(prediction1, axis=1)[0]  # Get predicted class index
        predicted_class = FRUITBOT_CLASSES[predicted_class_idx]  # Convert to class name
        
        return JSONResponse(content={"prediction": predicted_class})
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/predict/recyclebot0")
async def predict_fruitbot0(file: UploadFile = File(...)):
        return JSONResponse(content={"error": "This model is hosted at another endpoint"}, status_code=400)

@app.get("/working")
async def working():
    return JSONResponse(content={"Status": "Working"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)