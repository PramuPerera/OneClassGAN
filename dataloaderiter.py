import os
import numpy as np


def loadPaths(opt):
    dataset = opt.dataset
    datapath = opt.datapath
    classes = opt.classes
    expname = opt.expname
    # read names of training classes
    text_file = open(dataset + "_trainlist.txt", "r")
    folders = text_file.readlines()
    text_file.close()
    folders = [i.split('\n', 1)[0] for i in folders]
    inclasspaths = []
    testclasspaths = []
    inclasslabels = []
    testclasslabels = []
    # if classes is set to a a value use it instead
    inclasses = list(folders)
    if classes != "":
        inclasses = [classes]
    print(inclasses)
    # first 50% of each image is treated as training. remainder is treated as testing
    for lbl, nclass in enumerate(inclasses):
        dirs = os.listdir(datapath + dataset + '/' + nclass)
        for nfile in range(int(len(dirs)/2)):
            inclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
            inclasslabels.append(lbl)
        for nfile in range(int(len(dirs)/2)+1, len(dirs)):
            testclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
            testclasslabels.append(lbl)
    text_file = open(dataset + "_novellist.txt", "r")
    folders = text_file.readlines()
    text_file.close()
    folders = [i.split('\n', 1)[0] for i in folders]
    cluttersize = int(round(len(testclasslabels)/len(folders)))
    for i in range(len(folders) ):
        dirs = os.listdir(datapath + dataset + '/' + folders[i])
    for nfile in dirs[0: cluttersize]:
            testclasspaths.append(datapath + dataset + '/' +folders[i] + '/' + nfile)
            testclasslabels.append(-1)
    # write test files and labels to external file for future testing
    text_file = open(dataset + "_" + expname + "_testlist.txt", "w")
    for fn, lbl in zip(testclasspaths, testclasslabels):
        text_file.write("%s %s\n" % (fn, str(lbl)))
    text_file.close()

    text_file = open(dataset + "_" + expname + "_trainlist.txt", "w")
    for fn, lbl in zip(inclasspaths, inclasslabels):
        text_file.write("%s %s\n" % (fn, str(lbl)))
    text_file.close()    
    return [inclasspaths, inclasslabels]

