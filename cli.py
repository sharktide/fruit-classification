import os
import keyboard
import cv2
import numpy as np
import time
print("[DEBUG] program started!")



reshape2 = True
CLASSES = ['Apple 10', 'Apple 11', 'Apple 12', 'Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19', 'Apple 5', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple Core 1', 'Apple Red Yellow 2', 'Apple worm 1', 'Banana 3', 'Beans 1', 'Blackberrie 1', 'Blackberrie 2', 'Blackberrie half rippen 1', 'Blackberrie not rippen 1', 'Cabbage red 1', 'Cactus fruit green 1', 'Cactus fruit red 1', 'Caju seed 1', 'Cherimoya 1', 'Cherry Wax not rippen 1', 'Cucumber 10', 'Cucumber 9', 'Gooseberry 1', 'Pistachio 1', 'Quince 2', 'Quince 3', 'Quince 4', 'Tomato 1', 'Tomato 5', 'apple_6', 'apple_braeburn_1', 'apple_crimson_snow_1', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith_1', 'apple_hit_1', 'apple_pink_lady_1', 'apple_red_1', 'apple_red_2', 'apple_red_3', 'apple_red_delicios_1', 'apple_red_yellow_1', 'apple_rotten_1', 'cabbage_white_1', 'carrot_1', 'cucumber_1', 'cucumber_3', 'eggplant_long_1', 'pear_1', 'pear_3', 'zucchini_1', 'zucchini_dark_1']

import tensorflow as tf
import tensorflowtools as tft


modelrequested = str(input('''Which model to use? FruitBot0 (High Accuracy, Recommended), FruitBot1 (Older). Type 0/1 to select.
    >>> '''))
if (modelrequested == "0") :
    try :
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot0", "tf_model.keras")
    except :
        tft.hftools.download_model_from_huggingface("sharktide", "fruitbot0", "tf_model.keras")
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot0", "tf_model.keras")
elif (modelrequested == "1") :
    try :
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot1", "tf_model.keras")
        reshape2 = False
    except :
        tft.hftools.download_model_from_huggingface("sharktide", "fruitbot1", "tf_model.keras")
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot1", "tf_model.keras")
        reshape2 = False
else :
    raise TypeError("ERROR: No model selected.")

# Show the model architecture
ensemble1.summary()

check = 0

#Ask for Image to test


if reshape2 :
    test_image = cv2.resize(cv2.imread(input(f'''Path to image:
                                        -->''')),  (240, 240))
    test_image = np.array(test_image).reshape(-1, 240, 240, 3)
else :
    test_image = cv2.resize(cv2.imread(input(f'''Path to image:
                                        -->''')),  (224, 224))
    test_image = np.array(test_image).reshape(-1, 224, 224, 3)

print(test_image.shape)

# Get predictions (probabilities)
preds_1 = ensemble1.predict(test_image)



# Weighted average of probabilities
final_preds = preds_1
print(CLASSES)
print(final_preds)

print()


# Get the class with the highest weighted average probability
final_class = (CLASSES[np.argmax(final_preds)])
print(final_class)