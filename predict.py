from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models import Patchcore

# Load the trained model
model = Patchcore(
    backbone="resnet18",
    layers=["layer3"], #"layer2", 
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
    pre_processor=True,
    post_processor=True,
    evaluator=True,
    visualizer=True,
)

# Prepare the dataset
dataset = Folder(
    name="custom_dataset",
    root="/home/jonathanyeh/anomalib/test",
    normal_dir="/home/jonathanyeh/anomalib/test/good",
    abnormal_dir="/home/jonathanyeh/anomalib/test/defect_V0",
    mask_dir="/home/jonathanyeh/anomalib/test/masks",
    num_workers=39,
)

# Initialize the engine
engine = Engine()

# Predict and evaluate
results = engine.predict(model=model, datamodule=dataset, ckpt_path="/home/jonathanyeh/anomalib/results/Patchcore/my_dataset/v0/weights/lightning/model.ckpt",)
metrics = engine.validate(model=model, datamodule=dataset,ckpt_path="/home/jonathanyeh/anomalib/results/Patchcore/my_dataset/v0/weights/lightning/model.ckpt",)


print(metrics)
