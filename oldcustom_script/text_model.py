import onnx

# Preprocessing: load the ONNX model
model_path = "C:/Users/1003380/anomalib/exported_model/padim/weights/onnx/model.onnx"
onnx_model = onnx.load(model_path)

# print(f"The model is:\n{onnx_model}")

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print(f"The model is invalid: {e}")
else:
    print("The model is valid!")