import numpy as np
from PIL import Image, ImageChops

file = "all_images.txt"

with open(file) as f:
    lines = f.readlines()
    paths = []
    for line in lines:
        items = line.split()
        paths.append(items[0])

#img2 = Image.open("C:/Users/Jiankun/Desktop/Alexnet/black.png")
#img2 = img2.convert("L")
#img2[100:, 100:] = 255
array = 255 * np.ones((256, 256), dtype='int8')

for i in range(140):
    array[i, :(140-i)] = 10


array1 = np.reshape(array, -1)

img2 = Image.new("L", (256, 256))
img2.putdata(array1)
#print(list(img2.getdata()))

for i in range(1):
    img1 = Image.open(paths[i])
    img1 = img1.convert("L")
    img1.show()
    #print(list(img1.getdata()))
    img2 = img2.convert("L")
    img = ImageChops.multiply(img1, img2)
    #img = img.convert("RGB")

    #print(list(img.getdata()))
    img.show()