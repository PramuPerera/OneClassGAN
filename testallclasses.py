import options
import vaetest
import numpy as np
import os
import random
random.seed(1000)
opt = options.test_options()
opt.istest = 1

text_file = open(opt.dataset + "_progress.txt", "w")
text_file.close()
#First read all classes one at a time and iterate through all
#text_file = open(opt.dataset + "_folderlist.txt", "r")
#folders = text_file.readlines()
#text_file.close()
#folders = [i.split('\n', 1)[0] for i in folders]

follist = range(0,301,10)
folders = range(0,10)
for classname in [8]:#folders:#8,8,8,8,8]: #[8,2,3,0,1,4,5,6,7,9]:#folders:
        filelisttext = open(opt.dataset+'_trainlist.txt', 'w')
	filelisttext.write(str(classname))
	filelisttext.close()
        filelisttext = open(opt.dataset+'_novellist.txt','w')
	novellist = list(set(folders)-set([classname]))
 	print(novellist)
        for novel in novellist:
	        filelisttext.write(str(novel)+'\n')
        filelisttext.close()

        epoch = []
        trainerr = []
        valerr =[]
        #os.system('python2 cvpriterAAC.py --epochs 301 --batch_size 512 --ndf 8  --ngf 64 --istest 0 --expname Mnist  --img_wd 61 --img_ht 61 --depth 3 --datapath ../../mnist_png/mnist_png/  --noisevar 0.2  --lambda1 50 --seed 1000 --append 0 --dataset Mnist  --latent '+str(opt.latent))
	res_file = open(opt.expname + "_validtest.txt", "r")
        results = res_file.readlines()
        res_file.close()
        results = [i.split('\n', 1)[0] for i in results]
        print(results)
        for line in results:
            temp = line.split(' ', 1)
	    #print(temp)
            epoch.append(temp[0])
            temp = temp[1].split(' ', 1)
            trainerr.append(float(temp[0]))
            valerr.append(float(temp[1]))
        valep = np.argmin(np.array(valerr))
        trainep = np.argmin(np.array(trainerr))
        opt.epochs =follist[ valep]
        roc_aucval = vaetest.main(opt)
        opt.epochs = follist[trainep]
        roc_auctrain = vaetest.main(opt)
    	text_file = open(opt.dataset + "_progress.txt", "a")
        text_file.write("%s %s %s %s %s %s %s %s %s %s\n" % (str(valerr[valep]), str(trainerr[trainep]), str(roc_aucval[0]),str(roc_auctrain[0]), str(roc_aucval[1]),str(roc_auctrain[1]), str(roc_aucval[2]),str(roc_auctrain[2]), str(roc_aucval[3]),str(roc_auctrain[3]  )))
        text_file.close()
