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
from anomalib.deploy import ExportType

# ------------------------------
# 1. Training Phase (Optional)
# ------------------------------
# If you already have a trained checkpoint, skip this section

# train_datamodule = FolderDataModule(
    # name="candle_dataset",
    # root=Path(r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle"),
    # normal_dir="train/good",
    # abnormal_dir=None,  # train is only normal images
    # mask_dir=r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle\ground_truth\bad",
    # task="segmentation",
    # image_size=(1024, 1024),
    # batch_size=4,
# )
# train_datamodule.setup()

# model = Padim(
#     layers=["layer2", "layer3"],
#     backbone="resnet50",
#     pre_trained=True,
#     n_features=256,
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




post_processor = PostProcessor(image_sensitivity=0.5, pixel_sensitivity=0.3)
model.post_processor = post_processor

engine = Engine(
    max_epochs=1,
    enable_checkpointing=True,
    default_root_dir="results",
)

# Train model (skip if you already have checkpoint)
# engine.fit(model=model, datamodule=train_datamodule)

# ------------------------------
# 2. Inference Phase
# ------------------------------
test_datamodule = FolderDataset(
    name="candle_dataset_test",
    root=Path(r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle"),
    normal_dir="test/good",
    abnormal_dir="test/bad",
    mask_dir="ground_truth/bad",
    # task="segmentation",
    # image_size=(1024, 1024),
    # batch_size=1,
)
# test_datamodule.setup()

start = time.time()
predictions = engine.predict(
    model=model,
    dataset=test_datamodule,#.test_dataloader(),
    ckpt_path=r"C:\Users\1003380\anomalib\model\patchcore_candle.ckpt"
)
end = time.time()

print(f"\nTotal inference time: {end-start:.3f}s")
print(f"Average per image: {(end-start)/len(predictions):.4f}s")

# ------------------------------
# 3. Metrics Evaluation
# ------------------------------
# image_auroc = AUROC(fields=("pred_score", "gt_label"))

# #For pixel-level AUROC (segmentation)
# pixel_auroc = AUROC(fields=("pred_mask", "gt_mask"))
# y_true, y_pred = [], []

# for batch in predictions:
#     image_auroc = AUROC(fields=("pred_score","gt_label"))
#     pixel_auroc = AUROC(fields=("pred_mask","gt_mask"))

#     # update both metrics in one loop
#     for batch in predictions:
#         image_auroc.update(batch)

#         batch.pred_mask = batch.anomaly_map
#         pixel_auroc.update(batch)


#     if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
#         gt_mask = batch.gt_mask.flatten().numpy()
#         pred_mask = batch.pred_mask.flatten().numpy()
#         pixel_auroc.update(pred_mask, gt_mask)

# image_auroc.update(y_pred, y_true)

# print("\n--- Benchmark Results ---")
# print(f"Image-level AUROC: {image_auroc.compute():.4f}")
# print(f"Pixel-level AUROC: {pixel_auroc.compute():.4f}")


# ------------------------------
# 3. Metrics Evaluation
# ------------------------------

image_auroc = AUROC(fields=("pred_score", "gt_label"))
# pixel_auroc = AUROC(fields=("pred_score", "gt_label"))  # we will feed raw tensors manually
from torchmetrics.classification import BinaryAUROC
pixel_auroc = BinaryAUROC()


for batch in predictions:
    image_auroc.update(batch)
    # pixel_auroc.update(batch)
    if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
        pred = batch.anomaly_map.float().flatten()
        gt = batch.gt_mask.long().flatten()
        pixel_auroc.update(pred, gt)

print("\n--- Benchmark Results ---")
# print(f"Image-level AUROC: {image_auroc.compute():.4f}")
# print(f"Pixel-level AUROC: {pixel_auroc.compute():.4f}")
print("Image AUROC:", image_auroc.compute().item())
print("Pixel AUROC:", pixel_auroc.compute().item())

# ------------------------------
# 4. Optional Export Phase
# ------------------------------
# engine.export(
    # model=model,
    # export_root=Path("exported_models"),
    # input_size=(1024, 1024),
    # export_type=ExportType.OPENVINO,
# )


"""
Padim: 

Total inference time: 21.597s
Average per image: 0.1080s

--- Benchmark Results ---
Image AUROC: 0.9087499976158142
Pixel AUROC: 0.9825594425201416





"""