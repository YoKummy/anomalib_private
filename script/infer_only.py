# Complete Pipeline Example for VisA Candle Dataset
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import time
import torch
import numpy as np

from anomalib.data import PredictDataset, FolderDataset
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore
from anomalib.post_processing import PostProcessor
from anomalib.metrics import AUROC

# model = Padim(
    # layers=["layer2", "layer3"],
    # backbone="resnet50",
    # pre_trained=True,
    # n_features=256,
# )
model = Patchcore(
    backbone="resnet50", # resnet18, wide_resnet50_2 resnet50
    layers=["layer2", "layer3"], # 
    pre_trained=True,
    coreset_sampling_ratio=0.10,
    num_neighbors=9,#9
    pre_processor=True,
    post_processor=True,
    evaluator=True,
    visualizer=True,
)
# 
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Using device:", device)
# model = model.to(device)



post_processor = PostProcessor(image_sensitivity=0.5, pixel_sensitivity=0.3)
model.post_processor = post_processor

engine = Engine(
    max_epochs=1,
    enable_checkpointing=True,
    default_root_dir="results",
    accelerator="cuda"
)

test_datamodule = FolderDataset(
    name="candle_dataset_test",
    root=Path(r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle"),
    normal_dir="test/good",
    abnormal_dir="test/bad",
    mask_dir="ground_truth/bad",
)

start = time.time()
predictions = engine.predict(
    model=model,
    dataset=test_datamodule,#.test_dataloader(),
    ckpt_path=r"C:\Users\1003380\anomalib\model\patchcore_candle.ckpt"
)
end = time.time()

print(f"\nTotal inference time: {end-start:.3f}s")
print(f"Average per image: {(end-start)/len(predictions):.4f}s")

image_auroc = AUROC(fields=("pred_score", "gt_label"))
from torchmetrics.classification import BinaryAUROC
pixel_auroc = BinaryAUROC()


for batch in predictions:
    image_auroc.update(batch)
    if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
        pred = batch.anomaly_map.float().flatten()
        gt = batch.gt_mask.long().flatten()
        pixel_auroc.update(pred, gt)

print("\n--- Benchmark Results ---")
print("Image AUROC:", image_auroc.compute().item())
print("Pixel AUROC:", pixel_auroc.compute().item())


""" 
Padim:
16.28 fps
Total inference time: 12.290s
Average per image: 0.0614s

--- Benchmark Results ---
Image AUROC: 0.9097499847412109
Pixel AUROC: 0.9825743436813354

Patchcore:
6.74 fps
Total inference time: 29.669s
Average per image: 0.1483s

--- Benchmark Results ---
Image AUROC: 0.9535000324249268
Pixel AUROC: 0.9906408786773682


"""
