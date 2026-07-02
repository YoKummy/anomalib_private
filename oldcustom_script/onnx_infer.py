# import onnxruntime as ort
# import numpy as np
# from PIL import Image

# # === Paths ===
# onnx_model_path = r"C:\Users\1003380\anomalib\exported_model\padim\weights\onnx\model.onnx"
# img_path = r""

# # === Load image ===
# img = Image.open(img_path).convert("RGB")   # ensure 3 channels
# img = img.resize((256, 256))
# img = np.array(img, dtype=np.float32) / 255.0
# img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
# img = img.transpose(2, 0, 1)   # HWC -> CHW
# img = np.expand_dims(img, axis=0)  # (1, 3, H, W)

# # === Load model ===
# sess = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# # === Run inference ===
# input_name = sess.get_inputs()[0].name
# outputs = sess.run(None, {input_name: img})

# print("✅ Inference done.")
# for i, o in enumerate(outputs):
#     print(f"Output {i}: shape={o.shape}, dtype={o.dtype}")

# # --- inspect content ---
# pred_score = outputs[0]
# pred_label = outputs[1]
# anomaly_map = outputs[2]
# pred_mask = outputs[3]

# print("\nValues:")
# print(f"pred_score: {pred_score}")
# print(f"pred_label: {pred_label}")
# print(f"anomaly_map mean={np.mean(anomaly_map):.4f}, max={np.max(anomaly_map):.4f}, min={np.min(anomaly_map):.4f}")
# print(f"pred_mask sum={np.sum(pred_mask)} (white pixels = anomaly count)")



import onnxruntime as ort
import cv2
import numpy as np

# === Paths ===
onnx_model_path = r"C:\Users\1003380\anomalib\exported_model\padim\weights\onnx\model.onnx"
img_path = r"C:\Users\1003380\anomalib\datasets\LGP\train\isolated\20250910_PC1_18_y1024-2048_x14848-15872.png"
# normal img:  C:\Users\1003380\anomalib\datasets\LGP\mega\1_y0-1024_x2560-3584.png
# anomaly img: C:\Users\1003380\anomalib\datasets\LGP\train\isolated\20250910_PC1_18_y1024-2048_x14848-15872.png
#iso: C:\Users\1003380\anomalib\datasets\LGP\train\isolated\20250910_PC3_037_y512-1536_x1536-2560.png
# === Load session (GPU if possible) ===
sess = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

# === Preprocess ===
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (1024, 1024))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)   # add batch dim
    return img

# === Run inference ===
img_tensor = preprocess(img_path)
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: img_tensor})

# === Extract outputs ===
pred_score, pred_label, anomaly_map, pred_mask = outputs
anomaly_map = anomaly_map[0, 0]

print("✅ Inference done.")
print(f"pred_score: {pred_score.flatten()[0]:.4f}")
print(f"pred_label: {pred_label.flatten()[0]}")
print(f"anomaly_map mean={anomaly_map.mean():.4f}, max={anomaly_map.max():.4f}, min={anomaly_map.min():.4f}")

# === Normalize anomaly map ===
anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

# === Auto threshold ===
# We use the 95th percentile — pixels above this are likely anomalous.
auto_threshold = np.percentile(anomaly_map, 99.5)
print(f"Auto threshold = {auto_threshold:.4f}")

# === Create binary mask ===
custom_mask = (anomaly_map >= auto_threshold).astype(np.uint8) * 255

# === Save outputs ===
orig = cv2.imread(img_path)
mask_resized = cv2.resize(custom_mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)
overlay = cv2.addWeighted(orig, 0.7, cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET), 0.3, 0)

cv2.imwrite("custom_mask.png", mask_resized)
cv2.imwrite("overlay.png", overlay)
print("🖼️  Saved custom_mask.png and overlay.png")
