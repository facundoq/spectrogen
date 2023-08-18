from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

folder = Path("/home/facundoq/data/datasets/spectrogram/scans2023/")

file = 10
files = list([x for x in folder.iterdir() if x.is_file()])

filepath = files[file]

print(f"Loading {filepath}")

image = imread(filepath)

plt.imshow(image,cmap="gray")
plt.pause(0)
