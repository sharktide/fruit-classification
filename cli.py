import os
import cv2
import numpy as np
import time
print("[DEBUG] program started!")



reshape2 = 240
CLASSES = ['Apple 10', 'Apple 11', 'Apple 12', 'Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19', 'Apple 5', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple Core 1', 'Apple Red Yellow 2', 'Apple worm 1', 'Banana 3', 'Beans 1', 'Blackberrie 1', 'Blackberrie 2', 'Blackberrie half rippen 1', 'Blackberrie not rippen 1', 'Cabbage red 1', 'Cactus fruit green 1', 'Cactus fruit red 1', 'Caju seed 1', 'Cherimoya 1', 'Cherry Wax not rippen 1', 'Cucumber 10', 'Cucumber 9', 'Gooseberry 1', 'Pistachio 1', 'Quince 2', 'Quince 3', 'Quince 4', 'Tomato 1', 'Tomato 5', 'apple_6', 'apple_braeburn_1', 'apple_crimson_snow_1', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith_1', 'apple_hit_1', 'apple_pink_lady_1', 'apple_red_1', 'apple_red_2', 'apple_red_3', 'apple_red_delicios_1', 'apple_red_yellow_1', 'apple_rotten_1', 'cabbage_white_1', 'carrot_1', 'cucumber_1', 'cucumber_3', 'eggplant_long_1', 'pear_1', 'pear_3', 'zucchini_1', 'zucchini_dark_1']
EXTCLASSES = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red', 'Apple Red Delicious', 'Apple Red Yellow', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Carambula', 'Cauliflower', 'Cherry', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach Flat', 'Pear', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']
CLASSES_IMP = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
import tensorflow as tf
import tensorflowtools as tft

classes_to_use = "CLASSES"

modelrequested = str(input('''Which model to use? FruitBot Expanded (36 classes, high accuracy) FruitBot0 (Improved FruitBot1), FruitBot1 (Older). Type 2/0/1 to select.
    >>> '''))
if (modelrequested == "0") :
    try:
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot0", "tf_model.keras")
        reshape2=240
    except:
        print('Failed to load model')
        tft.hftools.download_model_from_huggingface("sharktide", "fruitbot0", "tf_model.keras")
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot0", "tf_model.keras")
elif (modelrequested == "1") :
    try :
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot1", "tf_model.keras")
        reshape2 = 224
    except :
        tft.hftools.download_model_from_huggingface("sharktide", "fruitbot1", "tf_model.keras")
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot1", "tf_model.keras")
        reshape2 = 224
elif (modelrequested == "2") :
    try :
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot-expanded", "tf_model.h5")
        reshape2 = 240
        classes_to_use = "CLASSES_IMP"
    except :
        tft.hftools.download_model_from_huggingface("sharktide", "fruitbot-expanded", "tf_model.h5")
        ensemble1 = tft.kerastools.load_from_hf_cache("sharktide", "fruitbot-expanded", "tf_model.h5")
        reshape2 = 240
        classes_to_use = "CLASSES_IMP"
else :
    raise TypeError("ERROR: No model selected.")

# Show the model architecture
ensemble1.summary()

check = 0

#Ask for Image to test


def full_classes():
    global test_image
    global ensemble1
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

def red_classes():
    global test_image
    global ensemble1
    print(test_image.shape)

    # Get predictions (probabilities)
    preds_1 = ensemble1.predict(test_image)



    # Weighted average of probabilities
    final_preds = preds_1
    if classes_to_use == "CLASSES_IMP":
        print(CLASSES_IMP)
        print(final_preds)

        print()


        # Get the class with the highest weighted average probability
        final_class = (CLASSES_IMP[np.argmax(final_preds)])
        print(final_class)
    else:
        print(CLASSES)
        print(final_preds)

        print()


        # Get the class with the highest weighted average probability
        final_class = (CLASSES[np.argmax(final_preds)])
        print(final_class)

if reshape2 == 240 :
    test_image = cv2.resize(cv2.imread(input(f'''Path to image:
                                        -->''')),  (240, 240))
    test_image = np.array(test_image).reshape(-1, 240, 240, 3)
    red_classes()
elif reshape2 == 224 :
    test_image = cv2.resize(cv2.imread(input(f'''Path to image:
                                        -->''')),  (224, 224))
    test_image = np.array(test_image).reshape(-1, 224, 224, 3)
    red_classes()
elif reshape2 == 100 :
    test_image = cv2.resize(cv2.imread(input(f'''Path to image:
                                        -->''')),  (100, 100))
    test_image = np.array(test_image).reshape(-1, 100, 100, 3)
    full_classes()

