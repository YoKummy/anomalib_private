from pathlib import Path
from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.models import Padim
from anomalib.post_processing import PostProcessor
from anomalib.data import FolderDataset
import time
import torch
from anomalib.metrics import AUROC, Evaluator
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

dataset = FolderDataset(
    name="candle_dataset",
    root=Path(r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle"),
    normal_dir="train/good",
    abnormal_dir="test/bad",         # or whatever your folder is called
    mask_dir="ground_truth",
    # task="segmentation",
    # image_size=(1024, 1024),
)
#"C:\Users\1003380\good"  "C:\Users\1003380\anomalib\datasets\LGP\train\good_small" C:\Users\1003380\anomalib\datasets\LGP\train\mini

folder = Path(r"C:\Users\1003380\spot-diff\VisA_pytorch\onecls\candle")
dicct = {f.name for f in folder.iterdir() if f.is_file()}
image_files = sorted([f for f in folder.iterdir() if f.is_file()])

#check point a
start = time.time()
predictions = engine.test(
    model=model,
    datamodule=dataset,
    ckpt_path=r"C:\Users\1003380\anomalib\model\candel.ckpt",
)

end = time.time()
t = round(end - start, 4)
print(f"Time elapsed: {end - start:.3f}  seconds")
print("Speed:", round(t/ 1000.0, 4), "seconds per image") #len(dicct), 4

print(predictions)