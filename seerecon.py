import options
import vaetest
import numpy as np
import os
import random
import mxnet as mx
import matplotlib.pyplot as plt
import mxnet.ndarray as nd
import visual
random.seed(1000)
opt = options.test_options()
opt.istest = 0

text_file = open(opt.dataset + "_progress.txt", "w")
text_file.close()
#First read all classes one at a time and iterate through all
#text_file = open(opt.dataset + "_folderlist.txt", "r")
#folders = text_file.readlines()
#text_file.close()
#folders = [i.split('\n', 1)[0] for i in folders]

follist = range(0,201,10)
folders = range(0,10)
for classname in [0]: #folders:
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
        #os.system('python2 cvpriterAAC.py --epochs 201 --batch_size 512 --ndf 8 --ngf 64 --istest 0 --expname grapesip64 --img_wd 61 --img_ht 61 --depth 3 --datapath ../mnist_png/mnist_png/ --noisevar 0.2 --lambda1 500 --seed 1000 --append 0 --dataset Mnist')
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
            trainerr.append(temp[0])
            valerr.append(temp[1])
        valep = np.argmin(np.array(valerr[5:len(valerr)]))
        trainep = np.argmin(np.array(trainerr[5:len(trainerr)]))
        #opt.epochs =follist[ valep]


    	ctx = mx.gpu() if opt.use_gpu else mx.cpu()



    	netEn,netDe, netD, netD2 = vaetest.set_network(opt.depth, ctx, 0, 0, opt.ndf, opt.ngf, opt.append)
    	netEn.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_En.params', ctx=ctx)
    	netDe.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_De.params', ctx=ctx)
    	netD.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D.params', ctx=ctx)
    	netD2.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D2.params', ctx=ctx)

	fakecode = nd.random_normal(loc=0, scale=1, shape=(16, 4096,1,1), ctx=ctx)
	out = netDe(fakecode)
        fake_img1 = nd.concat(out[0],out[1], out[2], out[3],dim=1)
	fake_img2 = nd.concat(out[7],out[6], out[5], out[4],dim=1)
	fake_img3 = nd.concat(out[8],out[9], out[10], out[11],dim=1)
	fake_img4 = nd.concat(out[15],out[14], out[13], out[12],dim=1)        
	fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img4, dim=2)
        #print(np.shape(fake_img))
        visual.visualize(fake_img)
        plt.savefig('outputs/fakes_'+opt.expname+'_.png')
