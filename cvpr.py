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
import dataloader as dload
import load_image
import visual
import models
import imagePool
from datetime import datetime
import time
import logging
import argparse
#logging.basicConfig()

def train_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", default="expce", help="Name of the experiment")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size per iteration")
    parser.add_argument("--epochs", default=1001, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--use_gpu", default=1, type=int,  help="1 to use GPU  ")
    parser.add_argument("--dataset", default="Caltech256",
                        help="Specify the training dataset  ")
    parser.add_argument("--lr", default="0.0002", type=float, help="Base learning rate")
    parser.add_argument("--ngf", default=64, type=int, help="Number of base filters")
    parser.add_argument("--beta1", default=0.5, type=float, help="Parameter for Adam")
    parser.add_argument("--lambda1", default=100, type=int, help="Weight of reconstruction loss")
    parser.add_argument("--pool_size", default=50, type=int, help="Number of pool for discriminator")
    parser.add_argument("--datapath", default='/users/pramudi/Documents/data/', help="Data path")
    parser.add_argument("--img_wd", default=256, type=int, help="Image width")
    parser.add_argument("--img_ht", default=256, type=int, help="Image height")
    parser.add_argument("--continueEpochFrom", default=-1,
                        help="Continue training from specified epoch")
    parser.add_argument("--graphvis", default=0,   help="1 to visualize the model")
    parser.add_argument("--noisevar", default=0.02, type=float, help="variance of noise added to input")
    parser.add_argument("--depth", default=4, type=int, help="Number of core layers in Generator/Discriminator")
    args = parser.parse_args()
    if args.use_gpu == 1:
        args.use_gpu = True
    else:
        args.use_gpu = False
    if args.graphvis == 1:
        args.graphvis = True
    else:
        args.graphvis = False
    
    return args

def set_network(depth, ctx, lr, beta1, ngf):
    # Pixel2pixel networks
    netG = models.CEGenerator(in_channels=3, n_layers=depth, ngf=ngf)  # UnetGenerator(in_channels=3, num_downs=8) #
    netD = models.Discriminator(in_channels=3, n_layers =depth, ngf=ngf)

    # Initialize parameters
    models.network_init(netG, ctx=ctx)
    models.network_init(netD, ctx=ctx)

    # trainer for the generator and the discriminator
    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    return netG, netD, trainerG, trainerD

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def train(pool_size, epochs, train_data, ctx, netG, netD, trainerG, trainerD, lambda1, batch_size):
        
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L1Loss()
    image_pool = imagePool.ImagePool(pool_size)
    metric = mx.metric.CustomMetric(facc)

    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)

    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)

            fake_out = netG(real_in)
            fake_concat = image_pool.query(fake_out)
            #fake_concat = image_pool.query(nd.concat(real_in, fake_out, dim=1))
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history images
                output = netD(fake_concat)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label, ], [output, ])

                # Train with real image
                real_concat = real_out
                output = netD(real_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD = (errD_real + errD_fake) * 0.5
                errD.backward()
                metric.update([real_label, ], [output, ])

            trainerD.step(batch.data[0].shape[0])

            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                fake_out = netG(real_in)
                fake_concat = fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1
                errG.backward()

            trainerG.step(batch.data[0].shape[0])

            # Print log infomation every ten batches
            if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info(
                    'discriminator loss = %f, generator loss = %f, binary training acc = %f at iter %d epoch %d'
                    % (nd.mean(errD).asscalar(),
                       nd.mean(errG).asscalar(), acc, iter, epoch))
            iter = iter + 1
            btic = time.time()

        name, acc = metric.get()
        metric.reset()
        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_G.params"
            netG.save_params(filename)
            # Visualize one generated image for each epoch
            fake_img = fake_out[0]
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
        #


def print_result():
    num_image = 4
    img_in_list, img_out_list = val_data.next().data
    for i in range(num_image):
        img_in = nd.expand_dims(img_in_list[i], axis=0)
        plt.subplot(2, 4, i + 1)
        visual.visualize(img_in[0])
        img_out = netG(img_in.as_in_context(ctx))
        plt.subplot(2, 4, i + 5)
        visual.visualize(img_out[0])
    plt.show()


opt = train_options()        
ctx = mx.gpu() if opt.use_gpu else mx.cpu()
inclasspaths = dload.loadPaths(opt.dataset, opt.datapath, opt.expname)
train_data, val_data = load_image.load_image(inclasspaths, opt.batch_size, opt.img_wd, opt.img_ht, opt.noisevar)
print('Data loading done.')
netG, netD, trainerG, trainerD = set_network(opt.depth, ctx, opt.lr, opt.beta1, opt.ngf)
if opt.graphvis:
    print(netG)
print('training')
train(opt.pool_size, opt.epochs, train_data, ctx, netG, netD, trainerG, trainerD, opt.lambda1, opt.batch_size)

