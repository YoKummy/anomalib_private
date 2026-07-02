import torch

ckpt_path = r"C:\Users\1003380\anomalib\model\padimgoodv2.ckpt"
ckpt = torch.load(ckpt_path)

# List all keys in the checkpoint
print(ckpt.keys())  # usually contains 'state_dict', maybe 'hyper_parameters', etc.

# Look inside the model state_dict
state_dict = ckpt['state_dict']
for k in state_dict.keys():
    print(k)