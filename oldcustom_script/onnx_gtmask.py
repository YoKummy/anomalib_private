# import onnxruntime as ort
# import cv2
# import numpy as np

# # Paths  C:\Users\1003380\anomalib\exported_model\padim\weights\onnx\padim.onnx  
# onnx_model_path = r"exported_model\padim\weights\onnx\model.onnx"
# img_path = r"total_defect/PC2_20250617_108_y0-1024_x10240-11264.png"
# # PC1_20250617_88_y1024-2048_x7680-8704  PC1_20250617_181_y0-1024_x8192-9216.png

# # Create ONNX Runtime session (use GPU if available)
# sess = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# # Preprocess image (match training)
# def preprocess(img):
#     img = cv2.resize(img, (256, 256))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype(np.float32) / 255.0
#     img = (img - np.array([0.485,0.456,0.406], dtype=np.float32)) / \
#           np.array([0.229,0.224,0.225], dtype=np.float32)
#     img = np.transpose(img, (2,0,1))  # HWC -> CHW
#     img = np.expand_dims(img, axis=0)  # add batch dimension
#     return img

# input_tensor = preprocess(cv2.imread(img_path))

# # Run inference
# input_name = sess.get_inputs()[0].name
# outputs = sess.run(None, {input_name: input_tensor})

# print("Output shapes:", [o.shape for o in outputs])


# import cv2
# import numpy as np

# # Use the last output as the anomaly map
# anomaly_map = outputs[-1][0,0].astype(np.float32)  # shape: (256, 256)

# # Normalize to 0-1
# anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

# # Threshold to create binary mask
# threshold = 0.5  # tweak as needed
# gt_mask = np.zeros_like(anomaly_map, dtype=np.uint8)
# gt_mask[anomaly_map >= threshold] = 255  # white = anomaly

# # Save the mask
# cv2.imwrite("example_gt_mask.png", gt_mask)

# orig_img = cv2.imread(img_path)
# gt_mask_resized = cv2.resize(gt_mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
# cv2.imwrite("example_gt_mask_resized.png", gt_mask_resized)


import torch
import onnxruntime as ort
import cv2
import numpy as np
import os
import glob
import csv
import time

# === Paths ===
onnx_model_path = r"C:\Users\1003380\anomalib\exported_model\padim\weights\onnx\model.onnx"
image_folder = r"C:\Users\1003380\anomalib\datasets\LGP\mega"
output_folder = r"C:\Users\1003380\anomalib\results\onnx_batch"

os.makedirs(output_folder, exist_ok=True)

# === ONNX session ===
sess = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
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
    # cv2.imwrite(os.path.join(output_folder, f"{base_name}_mask.png"), mask_resized)
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
