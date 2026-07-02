from pathlib import Path
from anomalib.models import Patchcore
from anomalib.models import EfficientAd
from anomalib.models import Fastflow
from anomalib.models import Padim
from anomalib.models import UniNet
from anomalib.models import Cflow
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
model = Fastflow(
    backbone= "resnet18",
    pre_trained= True,
    flow_steps= 8,
    conv3x3_only= False,
    hidden_ratio= 1.0,
) #X can't go to the next epochs needing gt_mask
 """
""" 
model = EfficientAd(
    teacher_out_channels= 384,
    model_size= "small",
    lr= 0.0001,
    weight_decay= 1.0e-05,
    padding= False,
    pad_maps= True,
) #X too slow only 1 batch
 """
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
    n_features= 256, #64ssssss
    
) #X too big, vram booming

""" 
model = Cflow(
    backbone= "resnet50",
    layers=["layer3", "layer4"],
    pre_trained= True,
    fiber_batch_size= 128,
    decoder= "freia-cflow",
    condition_vector= 128,
    coupling_blocks= 8,
    clamp_alpha= 1.9,
    permute_soft= False,
    lr= 0.0001,

)
 """

engine = Engine( enable_progress_bar=True, accelerator="cuda",enable_checkpointing=True, enable_model_summary=True, devices=[1])
engine.fit(datamodule=datamodule, model=model,)
#max_epochs=1,
# model = EfficientAd.load_from_checkpoint("model.ckpt")
# engine = Engine(accelerator = "gpu")
#export_path = Path("./exported_model/patchcorev2")
#export_path.mkdir(exist_ok=True, parents=True)

#engine.export(model=model, export_root=export_path, export_type="onnx")
