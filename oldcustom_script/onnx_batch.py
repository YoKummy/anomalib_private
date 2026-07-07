import torch
import onnxruntime as ort
import numpy as np
import cv2
import time
import os
from glob import glob
from pathlib import Path
###### This one is for BATCH inference 
# ---------------- CONFIG ----------------
model_path = r"C:\Users\1003380\anomalib\exported_models\weights\onnx\efficientad.onnx"
image_folder = r"C:\Users\1003380\anomalib\dataset\good_small\train\good"
img_size = (1024,1024)
batch_size = 8
WARMUP_BATCHES = 5
PRINT_EACH_BATCH = False
PREFER_CUDA = True
# ----------------------------------------


def preload_nvidia_dlls() -> None:
    """Add common CUDA/cuDNN DLL folders to the process search path on Windows."""
    if os.name != "nt":
        return

    dll_dirs: list[Path] = []
    env_root = None

    # Prefer directories from the current Python environment.
    try:
        import sys

        env_root = Path(sys.executable).resolve().parent
        site_packages = env_root / "Lib" / "site-packages"
        nvidia_root = site_packages / "nvidia"
        if nvidia_root.exists():
            dll_dirs.extend(p for p in nvidia_root.glob("*/bin") if p.is_dir())

        env_library_bin = env_root / "Library" / "bin"
        if env_library_bin.exists():
            dll_dirs.append(env_library_bin)
    except Exception:
        pass

    if env_root and (env_root / "Library" / "bin").exists():
        dll_dirs.append(env_root / "Library" / "bin")

    for dll_dir in dll_dirs:
        os.add_dll_directory(str(dll_dir))


preload_nvidia_dlls()
ort.preload_dlls()

# Load ONNX model
print("Available providers:", ort.get_available_providers())
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

cuda_provider_options = {
    "cudnn_conv_algo_search": "HEURISTIC",
    "do_copy_in_default_stream": "1",
}

providers = ["CPUExecutionProvider"]
if PREFER_CUDA:
    providers = [("CUDAExecutionProvider", cuda_provider_options), "CPUExecutionProvider"]

sess = ort.InferenceSession(
    model_path,
    sess_options=session_options,
    providers=providers,
)
print("Session providers:", sess.get_providers())

active_provider = sess.get_providers()[0] if sess.get_providers() else "UNKNOWN"
if active_provider != "CUDAExecutionProvider":
    print(f"WARNING: CUDAExecutionProvider is not active. Falling back to {active_provider}.")

input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]

# Gather images
image_paths = sorted(glob(os.path.join(image_folder, "*.*")))
print(f"Found {len(image_paths)} images.")

if not image_paths:
    raise FileNotFoundError(f"No images found in: {image_folder}")

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    return img

# Preprocess all images
imgs = [preprocess(p) for p in image_paths]
imgs = np.stack(imgs)
print(f"Batch shape: {imgs.shape}")

# ---------------- BOX EXTRACTION FUNCTION ----------------
def anomaly_to_boxes(anomaly_map, threshold=0.5, min_area=50):
    amap = anomaly_map.astype(np.float32)
    amap = (amap - amap.min()) / (amap.max() - amap.min() + 1e-6)
    mask = (amap > threshold).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x+w, y+h))
    return boxes
# ---------------------------------------------------------


# Inference in batches
results = []

# Warm-up runs stabilize CUDA kernels and produce more reliable timing.
for i in range(0, min(len(imgs), WARMUP_BATCHES * batch_size), batch_size):
    warmup_batch = imgs[i:i + batch_size]
    sess.run(output_names, {input_name: warmup_batch})

start_time = time.time()

for i in range(0, len(imgs), batch_size):
    batch = imgs[i:i + batch_size]
    batch_start = time.time()
    outputs = sess.run(output_names, {input_name: batch})
    batch_time = time.time() - batch_start

    if PRINT_EACH_BATCH:
        print(f"Batch {i//batch_size+1} done in {batch_time:.4f}s")
    results.append(outputs)

end_time = time.time()
total_time = end_time - start_time
avg_time_per_img = total_time / len(image_paths)

print("\n✅ Inference complete.")
print(f"Total time: {total_time:.3f}s for {len(image_paths)} images")
print(f"Average time per image: {avg_time_per_img:.4f}s")

# Flatten batch results → per image
flat_results = []
for batch_outputs in results:
    for b in range(batch_outputs[0].shape[0]):
        flat_results.append([o[b] for o in batch_outputs])


# ---------- PRINT EXAMPLE (your original feature kept) ----------
if flat_results:
    pred_score, pred_label, anomaly_map, pred_mask = flat_results[0]
    print(f"\nExample output for first image:")
    print(f"pred_score={pred_score}, pred_label={pred_label}")
    print(f"anomaly_map mean={anomaly_map.mean():.4f}, max={anomaly_map.max():.4f}")


# ---------- 🔥 NEW PART: Extract boxes for ALL images ----------
print("\n📦 Detected anomaly boxes per image:\n")

for idx, (pred_score, pred_label, anomaly_map, pred_mask) in enumerate(flat_results):
    amap = anomaly_map[0]  # shape [H, W]
    boxes = anomaly_to_boxes(amap, threshold=0.5)

    print(f"Image {idx}: score={float(pred_score):.3f}  boxes={boxes}")

    """
    Padim at 256, 256
    ✅ Inference complete.
    Total time: 49.532s for 2264 images
    Average time per image: 0.0219s
    45 fps
    Example output for first image:
    pred_score=[0.6709006], pred_label=[ True]
    anomaly_map mean=0.5926, max=0.6671

    512, 512
    ✅ Inference complete.
    Total time: 50.622s for 2264 images
    Average time per image: 0.0224s
    44 fps
    Example output for first image:
    pred_score=[0.62797904], pred_label=[ True]
    anomaly_map mean=0.5420, max=0.6252

    """