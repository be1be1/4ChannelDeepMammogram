import numpy as np
from PIL import Image, ImageChops, ImageEnhance

file = "all_images_old.txt"

with open(file) as f:
    lines = f.readlines()
    paths = []
    for line in lines:
        #items = line.split()
        paths.append(line)


shuffled_paths = np.random.permutation(paths)


testfiles = open("train_files.txt", "w")
valfiles = open("val_files.txt", "w")


for i in range(len(shuffled_paths)):

    if i > int(0.8 * len(shuffled_paths)):

        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '.jpg', str(shuffled_paths[i][-3:-2])))
        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_a.jpg', str(shuffled_paths[i][-3:-2])))
        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_b.jpg', str(shuffled_paths[i][-3:-2])))
        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_c.jpg', str(shuffled_paths[i][-3:-2])))
        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_d.jpg', str(shuffled_paths[i][-3:-2])))
        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_e.jpg', str(shuffled_paths[i][-3:-2])))
        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_f.jpg', str(shuffled_paths[i][-3:-2])))
        valfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_g.jpg', str(shuffled_paths[i][-3:-2])))

    else:

        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '.jpg', str(shuffled_paths[i][-3:-2])))
        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_a.jpg', str(shuffled_paths[i][-3:-2])))
        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_b.jpg', str(shuffled_paths[i][-3:-2])))
        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_c.jpg', str(shuffled_paths[i][-3:-2])))
        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_d.jpg', str(shuffled_paths[i][-3:-2])))
        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_e.jpg', str(shuffled_paths[i][-3:-2])))
        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_f.jpg', str(shuffled_paths[i][-3:-2])))
        testfiles.write('{} {}\n'.format('/guest/z/zhan4554/Desktop' + str(shuffled_paths[i][2:-8]) + '_g.jpg', str(shuffled_paths[i][-3:-2])))

testfiles.close()
valfiles.close()
