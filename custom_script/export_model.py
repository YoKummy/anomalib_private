from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.data import Folder
datamodule = Folder(
    name="my_dataset",
    root=(r"C:\Users\1003380\anomalib\datasets\LGP"), #/home/jonathanyeh/anomalib/datasets/LGP_new/  /home/jonathanyeh/anomalib/datasets
    normal_dir="mega",  # new_mini Subfolder containing normal images   CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    #abnormal_dir="anomaly",  # Subfolder containing anomalous images
    train_batch_size=8,
    # eval_batch_size=2,
    num_workers=39,
    
)


model = Padim(
    layers=['layer2', 'layer3'],
    backbone= 'resnet50',
    pre_trained= True,
    n_features= 256, 
)

engine = Engine()
""" 
model.to_onnx(
    "./exports",
    input_size= (224, 224),
    opset_version = 11,
    do_constant_folding = True,
    verifying = True
)
 """


onnx_model = engine.export(
    datamodule=datamodule,
    model = model,
    export_type="onnx",
    ckpt_path= r"C:\Users\1003380\anomalib\model\padimgoodv2.ckpt",
    export_root= r"C:\Users\1003380\anomalib\exported_model\padim"
)

print("Done")