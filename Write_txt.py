import os
import numpy as np

def ReadImages(channel):

    string = 'LEFT_' if channel[-1] == 'L' else 'RIGHT_'

    # path to the images
    path = "/guest/z/zhan4554/Desktop/Data_raw/" + string + channel[:len(channel)-1]

    paths_lst = [[],[],[],[]]
    for label in ['1','2','3','4']:
        for root, dirs, files in os.walk(path+'/'+label):
            for file in files:

                file_path = os.path.join(root, file).replace('\\','/')
                paths_lst[int(label)-1].append(file_path)

    return paths_lst


CCL = ReadImages('CCL')
CCR = ReadImages('CCR')
MLOL = ReadImages('MLOL')
MLOR = ReadImages('MLOR')


f1 = open("train_4_ch.txt", 'w')  # 5 cols, 4 channels + 1 label per line
f2 = open("train_1_ch.txt", 'w')  # 2 cols, 1 channel + 1 label per line
f3 = open("val_4_ch.txt", 'w')
f4 = open("val_1_ch.txt", 'w')

for i in range(4):

    # split data into training set and validation set by index
    length = len(CCL[i])
    val_idx = np.random.choice(np.arange(length), int(0.2 * length), replace=False)

    for j in range(length):

        if j not in val_idx:

            f1.write('{} {} {} {} {}\n'.format(CCL[i][j], CCR[i][j], MLOL[i][j], MLOR[i][j], str(i + 1)))

            f2.write('{} {}\n'.format(CCL[i][j], str(i + 1)))
            f2.write('{} {}\n'.format(CCR[i][j], str(i + 1)))
            f2.write('{} {}\n'.format(MLOL[i][j], str(i + 1)))
            f2.write('{} {}\n'.format(MLOR[i][j], str(i + 1)))

        else:

            f3.write('{} {} {} {} {}\n'.format(CCL[i][j], CCR[i][j], MLOL[i][j], MLOR[i][j], str(i + 1)))

            f4.write('{} {}\n'.format(CCL[i][j], str(i + 1)))
            f4.write('{} {}\n'.format(CCR[i][j], str(i + 1)))
            f4.write('{} {}\n'.format(MLOL[i][j], str(i + 1)))
            f4.write('{} {}\n'.format(MLOR[i][j], str(i + 1)))



f1.close()
f2.close()
f3.close()
f4.close()
