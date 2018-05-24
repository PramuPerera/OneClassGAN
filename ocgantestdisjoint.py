#https://github.com/zackchase/mxnet-the-straight-dope/blob/master/chapter14_generative-adversarial-networks/pixel2pixel.ipynb
from __future__ import print_function
import os
import matplotlib as mpl
import tarfile
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet.gluon import nn, utils
from mxnet.gluon.nn import Dense, Activation, Conv2D, Conv2DTranspose, \
    BatchNorm, LeakyReLU, Flatten, HybridSequential, HybridBlock, Dropout
from mxnet import autograd
import numpy as np
from random import shuffle
from sklearn.metrics import roc_curve, auc
import load_image
import models
from datetime import datetime
import time
import logging



import argparse
#logging.basicConfig()

def test_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", default="expce", help="Name of the experiment")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per iteration")
    parser.add_argument("--epochs", default=1000, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--use_gpu", default=1, type=int,  help="1 to use GPU  ")
    parser.add_argument("--dataset", default="Caltech256",
                        help="Specify the training dataset  ")
    parser.add_argument("--ngf", default=64, type=int, help="Number of base filters")
    parser.add_argument("--datapath", default='/users/pramudi/Documents/data/', help="Data path")
    parser.add_argument("--img_wd", default=256, type=int, help="Image width")
    parser.add_argument("--img_ht", default=256, type=int, help="Image height")
    parser.add_argument("--depth", default=4, type=int, help="Number of core layers in Generator/Discriminator")
    args = parser.parse_args()
    if args.use_gpu == 1:
        args.use_gpu = True
    else:
        args.use_gpu = False
    
    return args


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()



def set_network(depth, ctx, lr, beta1, ndf):
    netG = models.CEGenerator(in_channels=3, n_layers=depth, istest=True, ndf=ndf)  # UnetGenerator(in_channels=3, num_downs=8) #
    netD = models.Discriminator(in_channels=3, n_layers=depth, istest=True, ndf=ndf)

    # Initialize parameters
    models.network_init(netG, ctx=ctx)
    models.network_init(netD, ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    return netG, netD, trainerG, trainerD

opt = test_options()
ctx = mx.gpu() if opt.use_gpu else mx.cpu()
testclasspaths = []
testclasslabels = []

with open(opt.dataset+"_"+opt.expname+"_testlist.txt" , 'r') as f:
    for line in f:
        testclasspaths.append(line.split(' ')[0])
        if int(line.split(' ')[1])==-1:
            testclasslabels.append(0)
        else:
            testclasslabels.append(1)

test_data = load_image.load_test_images(testclasspaths,testclasslabels,opt.batch_size, opt.img_wd, opt.img_ht, ctx=ctx)
netG, netD, trainerG, trainerD = set_network(opt.depth, ctx, 0, 0, opt.ngf)
netG.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_G.params', ctx=ctx)
netD.load_params('checkpoints/'+opt.expname+'_'+str(opt.epochs)+'_D.params', ctx=ctx)
lbllist = [];
scorelist = [];
test_data.reset()
count = 0
for batch in (test_data):
    count+=1
    real_in = batch.data[0].as_in_context(ctx)
    real_out = batch.data[1].as_in_context(ctx)
    lbls = batch.label[0].as_in_context(ctx)
    out = (netG(real_in))
    real_concat = real_in
    #real_concat = nd.concat(out, out, dim=1)
    output = netD(real_out)#netD(real_concat)
    output = nd.mean(output, (1, 3, 2)).asnumpy()
    lbllist = lbllist+list(lbls.asnumpy())
    scorelist = scorelist+list(output)
    #visualize(out[0,:,:,:])
    #plt.savefig('outputs/testnet_T' + str(count) + '.png'))
fpr, tpr, _ = roc_curve(lbllist, scorelist, 1)
roc_auc = auc(fpr, tpr)
print(roc_auc)
