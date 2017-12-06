import glob
from PIL import Image, ImageOps

img = Image.open('C:/Data/LEFT_MLO/3/A_0443_1_LEFT_MLO.jpg')
img.show()
img1 = img.convert("RGB")
img1.show()
