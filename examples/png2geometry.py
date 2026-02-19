from gprmaxui.utils import  png2geometry
from pathlib import Path

if __name__ == "__main__":
    for image_file in Path("root_images").glob("*.png"):
        png2geometry(image_file, dxdydz=(0.002, 0.002, 0.002), scale=0.07)
