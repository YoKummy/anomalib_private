from pathlib import Path
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor
from anomalib.data import FolderDataModule, FolderDataset
from anomalib.metrics import AUROC
import time
import numpy as np

# ------------------------------
# Model setup
# ------------------------------
post_processor = PostProcessor(
    image_sensitivity=0.5,
    pixel_sensitivity=0.3,
)

model = Padim(
    layers=['layer2', 'layer3'],
    backbone='resnet50',
    pre_trained=True,
    n_features=256
)
model.post_processor = post_processor

engine = Engine()

# ------------------------------
# Dataset setup
# ------------------------------
test_dataset = FolderDataset(
    name="candle_test",
    path=Path(r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle\test"),
    gt_path=Path(r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle\ground_truth\bad"),
    # task="segmentation"
)

# ------------------------------
# Predict
# ------------------------------
start_time = time.time()
predictions = engine.predict(
    model=model,
    dataset=test_dataset,
    ckpt_path=r"C:\Users\1003380\anomalib\model\candel.ckpt"
)
end_time = time.time()
elapsed = end_time - start_time
print(f"Total inference time: {elapsed:.3f}s")
print(f"Speed per image: {elapsed/len(predictions):.4f}s")

# ------------------------------
# Evaluation
# ------------------------------
# Image-level AUROC
image_auroc = AUROC()
y_true = []
y_pred = []

# Pixel-level AUROC
pixel_auroc = AUROC()
for batch in predictions:
    # image-level
    y_true.append(int(batch.gt_label))
    y_pred.append(float(batch.pred_score))
    
    # pixel-level
    if hasattr(batch, "gt_mask") and batch.gt_mask is not None:
        gt_mask = batch.gt_mask.flatten().numpy()
        pred_mask = batch.pred_mask.flatten().numpy()
        pixel_auroc.update(pred_mask, gt_mask)

# Compute metrics
image_auroc.update(y_pred, y_true)
final_image_auroc = image_auroc.compute()
final_pixel_auroc = pixel_auroc.compute()

print(f"Image-level AUROC: {final_image_auroc:.4f}")
print(f"Pixel-level AUROC: {final_pixel_auroc:.4f}")
