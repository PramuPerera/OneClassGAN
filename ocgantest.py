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
epochs = 200
batch_size = 10
use_gpu = True
ctx = mx.gpu() if use_gpu else mx.cpu()
lr = 0.0002
beta1 = 0.5
lambda1 = 100
pool_size = 50
datapath = '/home/labuser/Documents/data/'
dataset = 'Caltech256'
expname = 'expAE'

# read names of classes; treat last class as clutter
text_file = open(dataset+"_folderlist.txt", "r")
folders = text_file.readlines()
text_file.close()
folders = [i.split('\n', 1)[0] for i in folders]
valid_folders = []
inclasspaths = []
testclasspaths = []
inclasslabels = []
testclasslabels = []
# randomly pick 3 classes making sure each has more than 150 images
for i in range(len(folders)-1):
    dirs = os.listdir(datapath + dataset + '/' + folders[i])
    if len(dirs) > 150:
        valid_folders.append(folders[i])
inclasses = np.random.permutation(np.arange(len(valid_folders)))[0:3]
inclasses = [valid_folders[i] for i in inclasses]
## Should load this from experimant ##
inclasses = ['092.grapes', '109.hot-tub', '148.mussels']
print(inclasses)

# first 150 of each image is treated as training. remainder is treated as testing
for lbl, nclass in enumerate(inclasses):
    dirs = os.listdir(datapath + dataset + '/' + nclass)
    for nfile in range(150):
        inclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
        inclasslabels.append(lbl)
    for nfile in range(151, len(dirs)):
        testclasspaths.append(datapath + dataset + '/' + nclass + '/' + dirs[nfile])
        testclasslabels.append(lbl)

# pick 50% of images from clutter class
cluttersize = int(round(len(testclasslabels)*0.5))
dirs = os.listdir(datapath + dataset + '/' + folders[-1])
for nfile in range(min(cluttersize, len(dirs))):
    testclasspaths.append(datapath + dataset + '/' + folders[-1] + '/' + dirs[nfile])
    testclasslabels.append(-1)
# write test files and labels to external file for future testing
text_file = open(dataset+"_"+expname+"_testlist.txt", "w")
for fn, lbl in zip(testclasspaths, testclasslabels):
    text_file.write("%s %s\n" % (fn, str(lbl)))
text_file.close()
img_wd = 256
img_ht = 256

def load_data(fnames, lbl, batch_size, is_reversed=False):
    img_in_list = []
    img_out_list = []
    shuffle(fnames)

    for img in fnames:
        img_arr = mx.image.imread(img).astype(np.float32)/127.5 - 1
        img_arr = mx.image.imresize(img_arr, img_wd, img_ht)
        # Crop input and output images
        croppedimg = mx.image.fixed_crop(img_arr, 0, 0, img_wd, img_ht)
        img_arr_in, img_arr_out = [croppedimg+mx.random.normal(0, 0.2, croppedimg.shape),
                                   croppedimg]
        img_arr_in, img_arr_out = [nd.transpose(img_arr_in, (2, 0, 1)),
                                   nd.transpose(img_arr_out, (2, 0, 1))]
        img_arr_in, img_arr_out = [img_arr_in.reshape((1,) + img_arr_in.shape),
                                   img_arr_out.reshape((1,) + img_arr_out.shape)]
        img_in_list.append(img_arr_out if is_reversed else img_arr_in)
        img_out_list.append(img_arr_in if is_reversed else img_arr_out)

    tempdata = [nd.concat(*img_in_list, dim=0), nd.concat(* img_out_list, dim=0)]
    templbl = nd.array(np.array(lbl), ctx=ctx)

    itertest = mx.io.NDArrayIter(data=tempdata, label=templbl,
                                  batch_size=batch_size)

    return itertest


test_data = load_data(testclasspaths,testclasslabels, batch_size)


def visualize(img_arr):
    plt.imshow(((img_arr.asnumpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')


class CEGenerator(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_bias=False):
        super(CEGenerator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 4
            padding = int(np.ceil((kernel_size - 1) / 2))
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(LeakyReLU(alpha=0.2))
            nf_mult = 2;
            nf_mult_prev = 1;

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = 2 ** n
                self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
                self.model.add(LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = 2 ** n_layers
            self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(LeakyReLU(alpha=0.2))

            # Decoder
            self.model.add(Conv2DTranspose(channels=ndf * nf_mult / 2, kernel_size=kernel_size, strides=1,
                                           padding=padding, in_channels=ndf * nf_mult,
                                           use_bias=use_bias))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult / 2))
            self.model.add(LeakyReLU(alpha=0.2))

            for n in range(1, n_layers):
                nf_mult = nf_mult / 2
                self.model.add(Conv2DTranspose(channels=ndf * nf_mult / 2, kernel_size=kernel_size, strides=2,
                                               padding=padding, in_channels=ndf * nf_mult,
                                               use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult / 2))
                self.model.add(LeakyReLU(alpha=0.2))

            self.model.add(Conv2DTranspose(channels=in_channels, kernel_size=kernel_size, strides=2,
                                           padding=padding, in_channels=ndf))
            self.model.add(LeakyReLU(alpha=0.2))

    def hybrid_forward(self, F, x):
        out = self.model(x)
        # print(out)
        return out




# Define Unet generator skip block
class UnetSkipUnit(HybridBlock):
    def __init__(self, inner_channels, outer_channels, inner_block=None, innermost=False, outermost=False,
                 use_dropout=False, use_bias=False):
        super(UnetSkipUnit, self).__init__()

        with self.name_scope():
            self.outermost = outermost
            en_conv = Conv2D(channels=inner_channels, kernel_size=4, strides=2, padding=1,
                             in_channels=outer_channels, use_bias=use_bias)
            en_relu = LeakyReLU(alpha=0.2)
            en_norm = BatchNorm(momentum=0.1, in_channels=inner_channels)
            de_relu = Activation(activation='relu')
            de_norm = BatchNorm(momentum=0.1, in_channels=outer_channels)

            if innermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels, use_bias=use_bias)
                encoder = [en_relu, en_conv]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + decoder
            elif outermost:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2)
                encoder = [en_conv]
                decoder = [de_relu, de_conv, Activation(activation='tanh')]
                model = encoder + [inner_block] + decoder
            else:
                de_conv = Conv2DTranspose(channels=outer_channels, kernel_size=4, strides=2, padding=1,
                                          in_channels=inner_channels * 2, use_bias=use_bias)
                encoder = [en_relu, en_conv, en_norm]
                decoder = [de_relu, de_conv, de_norm]
                model = encoder + [inner_block] + decoder
            if use_dropout:
                model += [Dropout(rate=0.5)]

            self.model = HybridSequential()
            with self.model.name_scope():
                for block in model:
                    self.model.add(block)

    def hybrid_forward(self, F, x):
        if self.outermost:
            return self.model(x)
        else:
            return F.concat(self.model(x), x, dim=1)


# Define Unet generator
class UnetGenerator(HybridBlock):
    def __init__(self, in_channels, num_downs, ngf=64, use_dropout=True):
        super(UnetGenerator, self).__init__()

        # Build unet generator structure
        unet = UnetSkipUnit(ngf * 8, ngf * 8, innermost=True)
        for _ in range(num_downs - 5):
            unet = UnetSkipUnit(ngf * 8, ngf * 8, unet, use_dropout=use_dropout)
        unet = UnetSkipUnit(ngf * 8, ngf * 4, unet)
        unet = UnetSkipUnit(ngf * 4, ngf * 2, unet)
        unet = UnetSkipUnit(ngf * 2, ngf * 1, unet)
        unet = UnetSkipUnit(ngf, in_channels, unet, outermost=True)

        with self.name_scope():
            self.model = unet

    def hybrid_forward(self, F, x):
        return self.model(x)


# Define the PatchGAN discriminator
class Discriminator(HybridBlock):
    def __init__(self, in_channels, ndf=64, n_layers=3, use_sigmoid=False, use_bias=False):
        super(Discriminator, self).__init__()

        with self.name_scope():
            self.model = HybridSequential()
            kernel_size = 4
            padding = int(np.ceil((kernel_size - 1) / 2))
            self.model.add(Conv2D(channels=ndf, kernel_size=kernel_size, strides=2,
                                  padding=padding, in_channels=in_channels))
            self.model.add(LeakyReLU(alpha=0.2))

            nf_mult = 1
            for n in range(1, n_layers):
                nf_mult_prev = nf_mult
                nf_mult = min(2 ** n, 8)
                self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=2,
                                      padding=padding, in_channels=ndf * nf_mult_prev,
                                      use_bias=use_bias))
                self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
                self.model.add(LeakyReLU(alpha=0.2))

            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n_layers, 8)
            self.model.add(Conv2D(channels=ndf * nf_mult, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult_prev,
                                  use_bias=use_bias))
            self.model.add(BatchNorm(momentum=0.1, in_channels=ndf * nf_mult))
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Conv2D(channels=1, kernel_size=kernel_size, strides=1,
                                  padding=padding, in_channels=ndf * nf_mult))
            if use_sigmoid:
                self.model.add(Activation(activation='sigmoid'))

    def hybrid_forward(self, F, x):
        out = self.model(x)
        # print(out)
        return out


def param_init(param):
    if param.name.find('conv') != -1:
        if param.name.find('weight') != -1:
            param.initialize(init=mx.init.Normal(0.02), ctx=ctx)
        else:
            param.initialize(init=mx.init.Zero(), ctx=ctx)
    elif param.name.find('batchnorm') != -1:
        param.initialize(init=mx.init.Zero(), ctx=ctx)
        # Initialize gamma from normal distribution with mean 1 and std 0.02
        if param.name.find('gamma') != -1:
            param.set_data(nd.random_normal(1, 0.02, param.data().shape))


def network_init(net):
    for param in net.collect_params().values():
        param_init(param)


def set_network():
    # Pixel2pixel networks
    netG = CEGenerator(in_channels=3)
    netD = Discriminator(in_channels=6)

    # Initialize parameters
    network_init(netG)
    network_init(netD)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    return netG, netD, trainerG, trainerD


# Loss
GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
L1_loss = gluon.loss.L1Loss()

netG, netD, trainerG, trainerD = set_network()
netG.load_params('checkpoints/testnet_190_G.params', ctx=ctx)
netD.load_params('checkpoints/testnet_190_D.params', ctx=ctx)
from datetime import datetime
import time
import logging

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

lbllist = [];
scorelist = [];
test_data.reset()
count = 0
for batch in (test_data):
    print(count)
    count+=1
    real_in = batch.data[0].as_in_context(ctx)
    real_out = batch.data[1].as_in_context(ctx)
    lbls = batch.label[0].as_in_context(ctx)
    out = (netG(real_in))
    real_concat = nd.concat(real_in, real_in, dim=1)
    #real_concat = nd.concat(out, out, dim=1)
    output = netD(real_concat)
    output = nd.mean(output, (1, 3, 2)).asnumpy()
    lbllist = lbllist+list(lbls.asnumpy())
    scorelist = scorelist+list(output)
    visualize(out[0,:,:,:])
    plt.savefig('outputs/testnet_T' + str(count) + '.png')

print((lbllist))
print((scorelist))
fpr, tpr, _ = roc_curve(lbllist, scorelist, 0)
roc_auc = auc(fpr, tpr)
print(roc_auc)