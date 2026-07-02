from pathlib import Path
from anomalib.models import Padim
from anomalib.data import Folder
from anomalib.engine import Engine
import torch
from pytorch_lightning.strategies import DDPStrategy

torch.set_float32_matmul_precision("medium")
datamodule = Folder(
    name="my_dataset",
    root=("/home/jonathanyeh/anomalib/datasets/LGP_new/"), #/home/jonathanyeh/anomalib/datasets/LGP_new/  /home/jonathanyeh/anomalib/datasets
    normal_dir="mega",  # new_mini Subfolder containing normal images   CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    #abnormal_dir="anomaly",  # Subfolder containing anomalous images
    train_batch_size=8,
    # eval_batch_size=2,
    num_workers=39,
    
)
"""
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
) #X too big
#model = model.to("cuda:0")
"""
model = Padim(
    layers=['layer2', 'layer3'],
    backbone= 'resnet50',
    pre_trained= True,
    n_features= 256,
)


engine = Engine( enable_progress_bar=True, accelerator="cuda",enable_checkpointing=True, enable_model_summary=True, devices=[1])
engine.fit(datamodule=datamodule, model=model,)

export_path = engine.export(model = model, export_type = ExportType.ONNX)
