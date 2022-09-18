from pathlib import Path
import os
import sys

if len(sys.argv) < 2:
    print(f"Usage: nerf rename_images <path/to/folder/full/of/images>")
    exit()

img_dir = Path(os.getcwd()) / sys.argv[1]

def is_probably_image(file_path: Path) -> bool:
    print(f"{file_path}")
    print(f"{file_path.is_file()}")
    return file_path.is_file() and file_path.suffix in [".png", ".jpg", ".jpeg", ".dng", ".tiff"]

img_paths = [img_path for img_path in img_dir.iterdir() if is_probably_image(img_path)]

i = 0
for item in img_paths:
    i += 1
    new_name = f"{i:04d}{item.suffix}"
    print(f"renaming {item} to {new_name}")
    os.rename(item, item.parent / new_name)

        