import onnxruntime as ort
import cv2
import numpy as np

# Paths  C:\Users\1003380\anomalib\exported_model\padim\weights\onnx\padim.onnx  
onnx_model_path = r"C:\Users\1003380\anomalib\results\weights\onnx\padimv2.onnx"
img_path = r"C:\Users\1003380\anomalib\datasets\LGP\mega\PC1_20250617_88_y1024-2048_x7680-8704.png"
# PC1_20250617_88_y1024-2048_x7680-8704  PC1_20250617_181_y0-1024_x8192-9216.png

# Create ONNX Runtime session (use GPU if available)
sess = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider'])

# Preprocess image (match training)
def preprocess(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485,0.456,0.406], dtype=np.float32)) / \
          np.array([0.229,0.224,0.225], dtype=np.float32)
    img = np.transpose(img, (2,0,1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # add batch dimension
    return img

input_tensor = preprocess(cv2.imread(img_path))

# Run inference
input_name = sess.get_inputs()[0].name
outputs = sess.run(None, {input_name: input_tensor})

print("Output shapes:", [o.shape for o in outputs])


import cv2
import numpy as np

# Use the last output as the anomaly map
anomaly_map = outputs[-1][0,0].astype(np.float32)  # shape: (256, 256)

# Normalize to 0-1
anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())

# Threshold to create binary mask
threshold = 0.5  # tweak as needed
gt_mask = np.zeros_like(anomaly_map, dtype=np.uint8)
gt_mask[anomaly_map >= threshold] = 255  # white = anomaly

# Save the mask
cv2.imwrite("example_gt_mask.png", gt_mask)

orig_img = cv2.imread(img_path)
gt_mask_resized = cv2.resize(gt_mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
cv2.imwrite("example_gt_mask_resized.png", gt_mask_resized)
