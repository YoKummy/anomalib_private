# Complete Unsupervised Pipeline: Training, Tiling, Inference, and Edge Export
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import time

# Modern Anomalib Core Components
from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import EfficientAd
from anomalib.metrics import AUROC
from anomalib.deploy import ExportType

# -------------------------------------------------------------------------
# 1. Configuration & Paths
# -------------------------------------------------------------------------
DATA_ROOT = Path(r"/home/jonathanyeh/anomalib/datasets/good_small/")
CHECKPOINT_DIR = Path("results/checkpoints/EfficientAd")
EXPORT_DIR = Path("exported_models")

# Define target high-resolution camera dimension
IMAGE_SIZE = (1024, 1024)

# -------------------------------------------------------------------------
# 2. Data Preparation (No Masks / Classification-Style Evaluation)
# -------------------------------------------------------------------------
datamodule = Folder(
    name="production_dataset",
    root=DATA_ROOT,
    normal_dir="train/good",           # Train with normal-only samples
    test_split_mode="from_dir",        # Use explicit test folders
    normal_test_dir="test/good",
    abnormal_dir="test/bad",
    train_batch_size=1,
    eval_batch_size=1,
)

# -------------------------------------------------------------------------
# 3. Model & Engine Initialization
# -------------------------------------------------------------------------
# EfficientAD is extremely fast and drops memory bank overhead completely
model = EfficientAd(model_size="small")

engine = Engine(
    # EfficientAD converges quickly on small normal datasets
    max_epochs=20,
    enable_checkpointing=True,
    default_root_dir="results",
    accelerator="auto",               # Automatically utilizes your Ada 6000s
    # EfficientAD handles best on single GPU per execution instance
    devices=1
)

# -------------------------------------------------------------------------
# 4. Execution Pipeline (Train / Load -> Predict -> Export)
# -------------------------------------------------------------------------
RUN_TRAINING = False   # Toggle off if you are reloading an existing weight set
RUN_EXPORT = True     # Master toggle for model export
RUN_EXPORT_ONNX = True
RUN_EXPORT_OPENVINO = True

if RUN_TRAINING:
    print("\n--- Starting Model Training ---")
    engine.fit(model=model, datamodule=datamodule)
    ckpt_path = None  # Uses the newly trained weights directly in the engine memory
else:
    print("\n--- Skipping Training: Loading Existing Checkpoint ---")
    ckpt_path = "/home/jonathanyeh/anomalib/script/results/EfficientAd/production_dataset/v0/weights/lightning/model.ckpt"

# --- Inference Phase ---
print("\n--- Executing Inference Pass ---")
start_time = time.time()
predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=ckpt_path
)
end_time = time.time()

# --- Performance Metrics Output ---
total_time = end_time - start_time
print(f"\nTotal inference time: {total_time:.3f}s")
print(f"Average per image: {total_time / len(predictions):.4f}s")

# -------------------------------------------------------------------------
# 5. Metrics Evaluation
# -------------------------------------------------------------------------
image_auroc = AUROC(fields=("pred_score", "gt_label"))

for batch in predictions:
    image_auroc.update(batch)

print("\n--- Benchmark Validation Results ---")
print("Image-level AUROC:", round(image_auroc.compute().item(), 4))

# -------------------------------------------------------------------------
# 6. Edge Optimization & Export
# -------------------------------------------------------------------------
if RUN_EXPORT:
    if RUN_EXPORT_ONNX:
        print(f"\n--- Exporting ONNX Model to: {EXPORT_DIR} ---")
        engine.export(
            model=model,
            export_root=EXPORT_DIR,
            model_file_name="efficientad",
            input_size=IMAGE_SIZE,
            export_type=ExportType.ONNX,
            ckpt_path=ckpt_path,
        )

    if RUN_EXPORT_OPENVINO:
        print(
            f"\n--- Compiling & Exporting OpenVINO Model to: {EXPORT_DIR} ---")
        engine.export(
            model=model,
            export_root=EXPORT_DIR,
            model_file_name="efficientad",
            input_size=IMAGE_SIZE,
            export_type=ExportType.OPENVINO,
            ckpt_path=ckpt_path,
        )

    print("Export complete.")
# git test
