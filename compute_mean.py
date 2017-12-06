import numpy as np
from PIL import Image, ImageChops, ImageEnhance

file = "all_images.txt"

with open(file) as f:
    lines = f.readlines()
    paths = []
    for line in lines:
        items = line.split()
        paths.append(items[0])

mean = 0.

for i in range(len(paths)):
    img = Image.open(paths[i])
    mean += np.mean(img)/len(paths)

print(mean)