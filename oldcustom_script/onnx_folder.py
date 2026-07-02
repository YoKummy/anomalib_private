import torch
import onnxruntime as ort
import cv2
import numpy as np
import os
import glob
import csv
import time
# This one is for 
# === Paths ===
onnx_model_path = r"C:\Users\1003380\anomalib\exported_model\padim\weights\onnx\model.onnx"
image_folder = r"C:\Users\1003380\anomalib\datasets\LGP\mega"
output_folder = r"C:\Users\1003380\anomalib\results\onnx_batch"

os.makedirs(output_folder, exist_ok=True)

# === ONNX session ===
sess = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider"])
input_name = sess.get_inputs()[0].name

# === Preprocess function ===
def preprocess(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (256, 256))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - np.array([0.485,0.456,0.406],dtype=np.float32)) / np.array([0.229,0.224,0.225],dtype=np.float32)
    img_chw = np.transpose(img_norm, (2,0,1))
    return np.expand_dims(img_chw, axis=0), img  # return original as well

# === CSV log ===
csv_path = os.path.join(output_folder, "batch_log.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["image_name", "pred_score", "anomaly_pixels"])
start = time.time()
# === Process all PNG/JPG images ===
image_paths = glob.glob(os.path.join(image_folder, "*.png")) + glob.glob(os.path.join(image_folder, "*.jpg"))

for img_path in image_paths:
    img_tensor, orig_img = preprocess(img_path)
    outputs = sess.run(None, {input_name: img_tensor})
    
    pred_score, pred_label, anomaly_map, pred_mask = outputs
    anomaly_map = anomaly_map[0, 0]
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    
    # Auto threshold
    threshold = np.percentile(anomaly_map, 99.5)
    mask = (anomaly_map >= threshold).astype(np.uint8) * 255
    mask_resized = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Overlay visualization
    overlay = cv2.addWeighted(orig_img, 0.7, cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET), 0.3, 0)
    
    # Save
    base_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), mask_resized)
    # cv2.imwrite(os.path.join(output_folder, f"{base_name}_overlay.png"), overlay)
    
    # Log
    anomaly_pixels = np.sum(mask_resized > 0)
    csv_writer.writerow([base_name, float(pred_score.flatten()[0]), int(anomaly_pixels)])
    print(f"✅ {base_name}: pred_score={pred_score.flatten()[0]:.4f}, anomaly_pixels={anomaly_pixels}")


end = time.time()
t = round(end - start, 4)
print(f"Time elapsed: {end - start:.3f}  seconds")
print("Speed:", round(t/len(image_folder), 4), "seconds per image")


csv_file.close()
print(f"\n🎯 All done! Results saved in {output_folder}")

# 1292.982 / 23900 = 0.0541 aka 20fps cpu
# 766 / 23900 = 0.0320 aka 31fps gpu
# 1787.752 / 23900 = 0.0748 aka 13fps generate both image
# 708.489 / 23900 = 0.0296 aka 34fps
#911.171 / 23900 = 0.0381 aka 26fps with batch size 16