import os
import pandas as pd
import glob
from utils import *

def generate_train(dirRoot):
    files = sorted(glob.glob(os.path.join(dirRoot, '*.jpg')))
    tot = len(files)
    trainFileList = []
    testFileList = []

    trainFileList = files[:int(tot * 0.8)]
    testFileList = files[int(tot * 0.8):]

    fileCSV = pd.DataFrame(trainFileList)
    # print(fileCSV)
    fileCSV.to_csv(dirRoot.replace('images', 'train.csv'), sep=',', index=False, header=False)

    fileCSV = pd.DataFrame(testFileList)
    # print(fileCSV)
    fileCSV.to_csv(dirRoot.replace('images', 'test.csv'), sep=',', index=False, header=False)

def generate_test(dirRoot):
    files = sorted(glob.glob(os.path.join(dirRoot, '*.jpg')))
    testFileList = []
    for fileName in files:
        testFileList.append(fileName)

    fileCSV = pd.DataFrame(testFileList)
    # print(fileCSV)
    fileCSV.to_csv(dirRoot.replace('images', 'test.csv'), sep=',', index=False, header=False)


if __name__ == '__main__':
    generate_train('Dataset/Baidu_Dewords/dehw_train_dataset/images')
    generate_test('Dataset/Baidu_Dewords/dehw_testB_dataset/images')
 