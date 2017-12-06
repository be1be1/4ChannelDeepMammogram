import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter
'''
def ReadImages(channel):

    string = 'LEFT_' if channel[-1] == 'L' else 'RIGHT_'

    # path to the images
    path = "C:/Data/" + string + channel[:len(channel)-1]

    paths_lst = [[],[],[],[]]
    for label in ['1','2','3','4']:
        for root, dirs, files in os.walk(path+'/'+label):
            for file in files:
                
                if file[0] == 'A':
                    file_path = os.path.join(root, file).replace('\\','/')
                    paths_lst[int(label)-1].append(file_path)
                
                file_path = os.path.join(root, file).replace('\\', '/')
                paths_lst[int(label) - 1].append(file_path)

    return paths_lst


CCL_path = ReadImages('CCL')
CCR_path = ReadImages('CCR')
MLOL_path = ReadImages('MLOL')
MLOR_path = ReadImages('MLOR')



array_L = 255 * np.ones((256, 256), dtype='int8')
array_R = 255 * np.ones((256, 256), dtype='int8')


for i in range(140):
    array_L[i, :(140-i)] = 17
    array_R[i, (116+i):] = 17

array_L = np.reshape(array_L, -1)
array_R = np.reshape(array_R, -1)


filter_L = Image.new("L", (256, 256))
filter_R = Image.new("L", (256, 256))
filter_L.putdata(array_L)
filter_R.putdata(array_R)
'''

file = "all_images_old.txt"

with open(file) as f:
    lines = f.readlines()
    paths = []
    for line in lines:
        items = line.split()
        paths.append(items[0])


for i in range(len(paths)):

    img = Image.open(paths[i])
    img_a = ImageChops.invert(img)
    img_b = img.filter(ImageFilter.DETAIL)
    img_c = img.filter(ImageFilter.EDGE_ENHANCE)

    img_a.save(str(paths[i][:-4]) + "_a.jpg")
    img_b.save(str(paths[i][:-4]) + "_b.jpg")
    img_c.save(str(paths[i][:-4]) + "_c.jpg")


'''
for i in range(4):

    length = len(CCL_path[i])

    for j in range(length):
        
        #if MLOL_path[i][j][19] == 'A' and MLOR_path[i][j][20] == 'A' \
         #       and CCL_path[i][j][18] == 'A' and CCR_path[i][j][19] == 'A':

        img_ccl = Image.open(CCL_path[i][j])
        img_ccr = Image.open(CCR_path[i][j])
        img_mlol = Image.open(MLOL_path[i][j])
        img_mlor = Image.open(MLOR_path[i][j])


        img_ccl = ImageChops.multiply(img_ccl.convert("L"), filter_L.convert("L"))
        img_ccr = ImageChops.multiply(img_ccr.convert("L"), filter_R.convert("L"))
        img_mlol = ImageChops.multiply(img_mlol.convert("L"), filter_L.convert("L"))
        img_mlor = ImageChops.multiply(img_mlor.convert("L"), filter_R.convert("L"))


        #pixels = img1.getdata()
        #print(list(pixels))
        img_ccl.save(CCL_path[i][j][:-4] + "_a.jpg")
        img_ccr.save(CCR_path[i][j][:-4] + "_a.jpg")
        img_mlol.save(MLOL_path[i][j][:-4] + "_a.jpg")
        img_mlor.save(MLOR_path[i][j][:-4] + "_a.jpg")



for i in range(4):

    length = len(CCL_path[i])

    for j in range(length):

        img_ccl = Image.open(CCL_path[i][j])
        img_ccr = Image.open(CCR_path[i][j])
        img_mlol = Image.open(MLOL_path[i][j])
        img_mlor = Image.open(MLOR_path[i][j])

        img_ccl = ImageChops.invert(img_ccl)
        img_ccr = ImageChops.invert(img_ccr)
        img_mlol = ImageChops.invert(img_mlol)
        img_mlor = ImageChops.invert(img_mlor)

        # pixels = img1.getdata()
        # print(list(pixels))
        img_ccl.save(CCL_path[i][j][:-4] + "_b.jpg")
        img_ccr.save(CCR_path[i][j][:-4] + "_b.jpg")
        img_mlol.save(MLOL_path[i][j][:-4] + "_b.jpg")
        img_mlor.save(MLOR_path[i][j][:-4] + "_b.jpg")


for i in range(4):

    length = len(CCL_path[i])

    for j in range(length):

        img_ccl = Image.open(CCL_path[i][j])
        img_ccr = Image.open(CCR_path[i][j])
        img_mlol = Image.open(MLOL_path[i][j])
        img_mlor = Image.open(MLOR_path[i][j])

        img_ccl = ImageEnhance.Sharpness(img_ccl).enhance(2.5)
        img_ccr = ImageEnhance.Sharpness(img_ccr).enhance(2.5)
        img_mlol = ImageEnhance.Sharpness(img_mlol).enhance(2.5)
        img_mlor = ImageEnhance.Sharpness(img_mlor).enhance(2.5)


        img_ccl.save(CCL_path[i][j][:-4] + "_c.jpg")
        img_ccr.save(CCR_path[i][j][:-4] + "_c.jpg")
        img_mlol.save(MLOL_path[i][j][:-4] + "_c.jpg")
        img_mlor.save(MLOR_path[i][j][:-4] + "_c.jpg")
'''