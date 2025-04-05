import customtkinter as ctk
import cv2
import numpy as np
import tensorflow as tf
import tensorflowtools as tft
from tkinter import filedialog
import threading  # For smooth camera operation
import time  # For simulating loading times

# Define classes
CLASSES = ['Apple 10', 'Apple 11', 'Apple 12', 'Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19', 
           'Apple 5', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple Core 1', 'Apple Red Yellow 2', 'Apple worm 1', 
           'Banana 3', 'Beans 1', 'Blackberrie 1', 'Blackberrie 2', 'Blackberrie half rippen 1', 
           'Blackberrie not rippen 1', 'Cabbage red 1', 'Cactus fruit green 1', 'Cactus fruit red 1', 'Caju seed 1', 
           'Cherimoya 1', 'Cherry Wax not rippen 1', 'Cucumber 10', 'Cucumber 9', 'Gooseberry 1', 'Pistachio 1', 
           'Quince 2', 'Quince 3', 'Quince 4', 'Tomato 1', 'Tomato 5', 'apple_6', 'apple_braeburn_1', 
           'apple_crimson_snow_1', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith_1', 
           'apple_hit_1', 'apple_pink_lady_1', 'apple_red_1', 'apple_red_2', 'apple_red_3', 'apple_red_delicios_1', 
           'apple_red_yellow_1', 'apple_rotten_1', 'cabbage_white_1', 'carrot_1', 'cucumber_1', 'cucumber_3', 
           'eggplant_long_1', 'pear_1', 'pear_3', 'zucchini_1', 'zucchini_dark_1']

# Initialize app
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("Fruit Classification App")
app.geometry("600x500")

# Global variables
file_path = None
image_from_camera = None
model = None
reshape_to_240 = True
frame_stack = []

# Frame Functions
def switch_frame(new_frame):
    if frame_stack:
        frame_stack[-1].pack_forget()
    new_frame.pack(expand=True, fill="both")
    frame_stack.append(new_frame)

def go_back():
    if len(frame_stack) > 1:
        frame_stack.pop().pack_forget()
        frame_stack[-1].pack(expand=True, fill="both")

# Page 1: Input Selection
def input_selection_page():
    frame = ctk.CTkFrame(app)

    def select_file():
        global file_path
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            file_label.configure(text=f"Selected File: {file_path}")
            next_button.configure(state="normal")

    def capture_image():
        global image_from_camera
        switch_frame(camera_page())
    
    title = ctk.CTkLabel(frame, text="Step 1: Select Input Method", font=("Arial", 18))
    title.pack(pady=20)

    file_button = ctk.CTkButton(frame, text="Upload File", command=select_file)
    file_button.pack(pady=10)

    camera_button = ctk.CTkButton(frame, text="Use Camera", command=capture_image)
    camera_button.pack(pady=10)

    file_label = ctk.CTkLabel(frame, text="No file selected.", wraplength=400)
    file_label.pack(pady=10)

    next_button = ctk.CTkButton(frame, text="Next", state="disabled", command=lambda: switch_frame(model_selection_page()))
    next_button.pack(pady=20)

    return frame

# Page 2: Camera Capture
import threading

# Page 2: Camera Capture with Live Preview
def camera_page():
    frame = ctk.CTkFrame(app)

    from PIL import Image, ImageTk

    def camera_preview():
        global previewing
        previewing = True
        cap = cv2.VideoCapture(0)  # Open the camera
        while previewing:
            ret, frame_cv = cap.read()
            if not ret:
                break
            # Convert the frame to RGB for Tkinter
            frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            # Convert RGB frame into PIL Image
            pil_image = Image.fromarray(frame_rgb)
            # Resize the image using Image.Resampling.LANCZOS
            pil_image_resized = pil_image.resize((400, 300), Image.Resampling.LANCZOS)
            # Convert to ImageTk format for customtkinter
            frame_tk = ImageTk.PhotoImage(pil_image_resized)

            # Update the live camera feed in the label
            live_camera_label.configure(image=frame_tk, text="")  # Clear the text
            live_camera_label.image = frame_tk

            app.update_idletasks()

        cap.release()

    def start_camera_preview():
        # Start a new thread for the live camera feed
        threading.Thread(target=camera_preview, daemon=True).start()

    def stop_camera_preview_and_capture():
        global image_from_camera, previewing
        previewing = False
        cap = cv2.VideoCapture(0)
        ret, frame_cv = cap.read()
        if ret:
            image_from_camera = frame_cv
            switch_frame(model_selection_page())
        else:
            error_label.configure(text="Failed to capture image.")
        cap.release()

    # GUI Elements
    title = ctk.CTkLabel(frame, text="Step 2: Capture Image", font=("Arial", 18))
    title.pack(pady=20)

    # Live camera preview placeholder
    live_camera_label = ctk.CTkLabel(frame, text="Starting camera...", width=400, height=300)
    live_camera_label.pack(pady=10)

    # Start live camera preview
    start_camera_preview()

    capture_button = ctk.CTkButton(frame, text="Capture Image", command=stop_camera_preview_and_capture)
    capture_button.pack(pady=20)

    error_label = ctk.CTkLabel(frame, text="", wraplength=400, fg_color="red")
    error_label.pack(pady=10)

    back_button = ctk.CTkButton(frame, text="Back", command=go_back)
    back_button.pack(pady=20)

    return frame


# Page 3: Model Selection
def model_selection_page():
    frame = ctk.CTkFrame(app)

    def load_model_with_loading(selected_model):
        switch_frame(loading_page("Loading Model..."))
        app.update_idletasks()
        load_model(selected_model)

    def load_model(selected_model):
        global model, reshape_to_240
        time.sleep(1)  # Simulate model loading time
        try:
            if selected_model == "FruitBot0":
                try:
                    model = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot0", "tf_model.keras")
                except:
                    tft.hftools.download_model_from_huggingface('sharktide', 'fruitbot0', 'tf_model.keras')
                    model = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot0", "tf_model.keras")
                reshape_to_240 = True
            elif selected_model == "FruitBot1":
                try:
                    model = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot1", "tf_model.keras")
                except:
                    tft.hftools.download_model_from_huggingface('sharktide', 'fruitbot1', 'tf_model.keras')
                    model = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot1", "tf_model.keras")
                reshape_to_240 = False
            switch_frame(predict_page())
        except Exception as e:
            switch_frame(error_page(f"Model loading failed: {e}"))

    title = ctk.CTkLabel(frame, text="Step 3: Select a Model", font=("Arial", 18))
    title.pack(pady=20)

    fruitbot0_button = ctk.CTkButton(frame, text="FruitBot0 (High Accuracy)", command=lambda: load_model_with_loading("FruitBot0"))
    fruitbot0_button.pack(pady=10)

    fruitbot1_button = ctk.CTkButton(frame, text="FruitBot1 (Older)", command=lambda: load_model_with_loading("FruitBot1"))
    fruitbot1_button.pack(pady=10)

    back_button = ctk.CTkButton(frame, text="Back", command=go_back)
    back_button.pack(pady=20)

    return frame

# Loading Page
def loading_page(message="Loading..."):
    frame = ctk.CTkFrame(app)
    loading_label = ctk.CTkLabel(frame, text=message, font=("Arial", 18))
    loading_label.pack(pady=20)
    return frame

# Prediction Page
def predict_page():
    frame = ctk.CTkFrame(app)

    def predict_with_loading():
        switch_frame(loading_page("Predicting..."))
        app.update_idletasks()
        predict()

    def predict():
        try:
            # Simulate prediction time
            time.sleep(2)

            # Preprocess image
            if file_path:
                image = cv2.imread(file_path)
            elif image_from_camera is not None:
                image = image_from_camera
            else:
                switch_frame(error_page("No image available for prediction."))
                return
            
            if reshape_to_240:
                image = cv2.resize(image, (240, 240)).reshape(-1, 240, 240, 3)
            else:
                image = cv2.resize(image, (224, 224)).reshape(-1, 224, 224, 3)
            
            # Predict
            preds = model.predict(image)
            final_class = CLASSES[np.argmax(preds)]
            switch_frame(result_page(f"Prediction: {final_class}"))
        except Exception as e:
            switch_frame(error_page(f"Prediction failed: {e}"))

    title = ctk.CTkLabel(frame, text="Step 4: Prediction", font=("Arial", 18))
    title.pack(pady=20)

    predict_button = ctk.CTkButton(frame, text="Predict", command=predict_with_loading)
    predict_button.pack(pady=20)

    return frame

# Result Page
def result_page(result_message):
    frame = ctk.CTkFrame(app)

    result_label = ctk.CTkLabel(frame, text=result_message, font=("Arial", 18))
    result_label.pack(pady=20)

    new_prediction_button = ctk.CTkButton(frame, text="Make New Prediction", command=lambda: switch_frame(input_selection_page()))
    new_prediction_button.pack(pady=20)

    return frame

def error_page(error_message):
    frame = ctk.CTkFrame(app)

    error_label = ctk.CTkLabel(frame, text=error_message, font=("Arial", 18), fg_color="red")
    error_label.pack(pady=20)

    back_button = ctk.CTkButton(frame, text="Back to Input Selection", command=lambda: switch_frame(input_selection_page()))
    back_button.pack(pady=20)

    return frame

# Initialize First Page
switch_frame(input_selection_page())

# Run App
app.mainloop()
