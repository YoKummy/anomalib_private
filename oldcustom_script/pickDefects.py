import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm

# --- CONFIG ---
input_dir = r"C:\Users\1003380\anomalib\results\Padim\latest\images\mega"        # folder containing 21k images
output_dir = r"C:\Users\1003380\anomalib\defects"        # folder to save images with red color
os.makedirs(output_dir, exist_ok=True)

# Red color range in HSV
# Red is tricky because it wraps around 0° and 180° hue
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# Threshold: how many pixels must be red to consider image “red”
RED_PIXEL_RATIO_THRESHOLD = 0.001  # 1% of pixels are red

# --- PROCESS LOOP ---
for filename in tqdm(os.listdir(input_dir)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        continue
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    red_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    red_ratio = red_pixels / total_pixels

    if red_ratio > RED_PIXEL_RATIO_THRESHOLD:
        shutil.copy(img_path, os.path.join(output_dir, filename))
