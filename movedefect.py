from pathlib import Path
import shutil

# Paths
heatmap_folder = Path("/home/jonathanyeh/anomalib/datasets/LGP_new/total_defect/")
raw_folder = Path("/home/jonathanyeh/anomalib/datasets/LGP_new/mega/")
destination_folder = Path("/home/jonathanyeh/anomalib/datasets/LGP_new/del_def/")
destination_folder.mkdir(exist_ok=True)

# Collect filenames from heatmap folder
heatmap_files = {f.name for f in heatmap_folder.iterdir() if f.is_file()}
count = 0
# Iterate raw images and move matching ones
for raw_file in raw_folder.iterdir():
    if raw_file.is_file() and raw_file.name in heatmap_files:
        shutil.move(str(raw_file), destination_folder / raw_file.name)
        print(f"Moved: {raw_file.name}")
        count += 1

print("Total file: ", count)
'''
folder = Path("/home/jonathanyeh/anomalib/datasets/LGP_new/isolated")
dict = {f.name for f in folder.iterdir() if f.is_file()}
print(len(dict))
'''