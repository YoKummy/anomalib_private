#MWE script to export models to ONNX and run inference.

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


""" import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.deploy import ExportType, OpenVINOInferencer

model = Patchcore()
engine = Engine()
datamodule = MVTecAD(category="transistor")

# Train the model
engine.fit(datamodule=datamodule, model=model)
results = engine.test(datamodule=datamodule, model=model)

# export to onnx
export_path = engine.export(model=model, export_type=ExportType.ONNX)
print(f"Model exported to: {export_path}")

# OpenVINOInferencer can be used for ONNX models as well
inferencer = OpenVINOInferencer(path=export_path, device="CPU")
# dummy image 
dummy_image = torch.randn(1, 3, 256, 256)
predictions = inferencer.predict(image=dummy_image)
print(f"Predictions Keys: {predictions.keys()}")
# Predictions Keys: ['image', 'gt_label', 'gt_mask', 'mask_path', 'anomaly_map', 'pred_score', 'pred_mask', 'pred_label', 'explanation', 'image_path'] """

import torch
from anomalib.models import Padim
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.deploy import ExportType, OpenVINOInferencer
datamodule = Folder(
    name="my_dataset",
    root=("/home/jonathanyeh/anomalib/datasets/LGP_new/"), #/home/jonathanyeh/anomalib/datasets/LGP_new/  /home/jonathanyeh/anomalib/datasets
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

model = engine.fit(
    datamodule=datamodule,
    model = model,
    ckpt_path= "/home/jonathanyeh/anomalib/results/Padim/my_dataset/v4/weights/lightning/model.ckpt",
)

results = engine.test(datamodule=datamodule, model=model)

export_path = engine.export(model=model, export_type=ExportType.ONNX)
print(f"Model exported to: {export_path}")

print("Done")

inferencer = OpenVINOInferencer(path=export_path, device="CUDA")
# dummy image 
dummy_image = torch.randn(1, 3, 256, 256)
predictions = inferencer.predict(image=dummy_image)
print(f"Predictions Keys: {predictions.keys()}")