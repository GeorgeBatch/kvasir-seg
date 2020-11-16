
import os
import time
import numpy as np
import tensorflow as tf
import cv2
import pandas as pd

# Set path to test dataset
TEST_DATASET_PATH = "/medico2020"
MASK_PATH = "/mask"

# Load Keras model
model = tf.keras.models.load_model("/submission/model.h5")

time_taken = []
for image_name in os.listdir(TEST_DATASET_PATH):

    # Load the test image
    image_path = os.path.join(TEST_DATASET_PATH, image_name)
    image = cv2.imread(image_path)
    H, W, _ = image.shape
    image = cv2.resize(image, (512, 512))
    image = np.expand_dims(image, axis=0)

    # Start time
    start_time = time.time()

    ## Prediction
    mask = model.predict(image)[0]

    # End timer
    end_time = time.time() - start_time

    time_taken.append(end_time)
    print("{} - {:.10f}".format(image_name, end_time))

    mask = mask > 0.5
    mask = mask.astype(np.float32)
    mask = mask * 255.0
    mask = cv2.resize(mask, (H, W))

    mask_path = os.path.join(MASK_PATH, image_name)
    cv2.imwrite(mask_path, mask)

mean_time_taken = np.mean(time_taken)
mean_fps = 1/mean_time_taken
print("Mean FPS: ", mean_fps)
