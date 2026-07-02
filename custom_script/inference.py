from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor
import time
import torch
from anomalib.metrics import AUROC, Evaluator
# -------------------------------
# 1. Initialize the model
# -------------------------------


post_processor = PostProcessor(
    image_sensitivity=0.5,
    pixel_sensitivity=0.3,
)
# pre_processor = Patchcore.configure_pre_processor(image_size=(1024, 1024))
""" model = Patchcore(
    backbone="resnet50", # resnet50 wide_resnet50_2  resnet18
    layers=["layer2", "layer3"], # 
    pre_trained=True,
    pre_processor=True,
    post_processor=post_processor,
    evaluator=True,
    visualizer=True,
) """

model = Padim(
    layers=['layer2', 'layer3'],
    backbone= 'resnet50',
    pre_trained= True,
    n_features= 256,
    
)
model.post_processor = post_processor
engine = Engine()

# -------------------------------
# 2. Prepare test data
# -------------------------------
dataset = PredictDataset(
    path=Path(r"C:\Users\1003380\anomalib\datasets\LGP\mega"), #  C:\Users\1003380\anomalib\datasets\LGP\train\good_small  C:\Users\1003380\anomalib\datasets\LGP\train\normal    C:\Users\1003380\anomalib\datasets\LGP\test\defect_V0
)

# ------------------------------- "C:\Users\1003380\good"  "C:\Users\1003380\anomalib\datasets\LGP\train\good_small" C:\Users\1003380\anomalib\datasets\LGP\train\mini
# 3. Get predictions
# -------------------------------

folder = Path(r"C:\Users\1003380\anomalib\datasets\LGP\mega")
dicct = {f.name for f in folder.iterdir() if f.is_file()}
image_files = sorted([f for f in folder.iterdir() if f.is_file()])

#check point a
start = time.time()
predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=r"C:\Users\1003380\anomalib\model\padimgoodv2.ckpt",
)
end = time.time()
t = round(end - start, 4)
#final check point -> wanting to see how long this line takes to run
print(f"Time elapsed: {end - start:.3f}  seconds")
print("Speed:", round(t/len(dicct), 4), "seconds per image")

""" pred_scores = torch.tensor([float(pred.pred_score) for pred in predictions])
gt_label = torch.tensor([0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 
                0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

auroc = AUROC(fields=["pred_scores", "gt_label"])

from torchmetrics import AUROC as TAUROC
metric = TAUROC(task="binary")
metric.update(pred_scores, gt_label)
print("Image-level AUROC: ", metric.compute().item())

threshold = torch.quantile(pred_scores, 0.95)  # top 5% of normal scores as cutoff
pred_labels = (pred_scores > threshold).int()
accuracy = (pred_labels == gt_label).sum().item() / len(gt_label)
print(f"Image-level Accuracy: {accuracy:.3f}")



for idx, pred in enumerate(predictions):
    print(f"Image {idx}: raw score = {pred.pred_score}") """

""" image_threshold = 0.5234  # choose based on sensitivity you want
predicted_labels = []

for idx, prediction in enumerate(predictions):
    anomaly_score = float(prediction.pred_score)
    is_anomalous = int(anomaly_score > image_threshold)
    predicted_labels.append(is_anomalous)

# Compute accuracy
correct = sum([p == gt for p, gt in zip(predicted_labels, ground_truth)])
accuracy = correct / len(ground_truth)
print(f"Image-level Accuracy: {accuracy:.3f}")
 """

""" 
image_threshold = 0.3
correct = 0
total = len(predictions)

for idx, prediction in enumerate(predictions):
    # prediction.pred_score is the image-level anomaly score
    anomaly_score = float(pred.pred_score)
    is_anomalous = int(anomaly_score > image_threshold)

    true_label = ground_truth[idx]

    if is_anomalous == true_label:
        correct += 1

    print(f"Image {image_files[idx].name}: Pred={is_anomalous}, True={true_label}")

accuracy = correct / total
print(f"\nImage-level Accuracy: {accuracy:.3f} ({correct}/{total})")
 """


""" 
v1_18: 71.714 / 555 = 0.129 gpu
v2_18: 75.71 / 555 = 0.136 gpu
v3_50: 128 / 555 = 0.230 gpu
wide50: 132.287 / 555 = 0.238 gpu

v1_18 :13.491 / 99 = 0.136
v2_18 : 13.447 / 99 = 0.135
v1_50 : 23.648 / 99 = 0.238
wide50v1 : 23.710 / 99 = 0.239
fixwide50 : 25.935 / 99 = 0.262


# -------------------------------
# 4. Process predictions with thresholds
# -------------------------------

# Image-level threshold: controls whether the image is flagged as anomalous
image_threshold = 0.99  # lower = more sensitive, higher = stricter

# Pixel-level threshold: controls which pixels are considered anomalous
pixel_threshold = 0.99  # lower = highlight more pixels

if predictions:
    for prediction in predictions:
        print(prediction.image_path)

        image_path = prediction.image_path
        anomaly_score = float(prediction.pred_score)
        is_anomalous = anomaly_score > image_threshold

        print(f"Image: {image_path}")
        print(f"Anomaly Score: {anomaly_score:.3f}")
        print(f"Threshold: {image_threshold:.2f} -> Is Anomalous: {is_anomalous}")

        # Generate a binary pixel-level mask from the anomaly map
        if hasattr(prediction, "anomaly_map") and prediction.anomaly_map is not None:
            anomaly_map = prediction.anomaly_map.squeeze()
            binary_mask = (anomaly_map > pixel_threshold).float()

            # Fix: take the first element from the list
            image_path_str = prediction.image_path[0]
            mask_path = Path("masks") / Path(image_path_str).name
            mask_path.parent.mkdir(exist_ok=True)
            binary_mask.numpy().tofile(mask_path)  # or np.save(mask_path, binary_mask.numpy())
 """