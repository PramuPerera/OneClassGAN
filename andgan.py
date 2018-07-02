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
import random
from random import shuffle
import dataloaderiter as dload
import load_image
import visual
import models
import imagePool
from datetime import datetime
import time
import logging
import argparse
import options


def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()


def trainadnov(opt, train_data, val_data, ctx, networks):

    netEn = networks[0]
    netDe = networks[1]
    netD = networks[2]
    netD2 = networks[3]
    trainerEn = networks[4]
    trainerDe = networks [5]
    trainerD =networks[6]
    trainerD2 = networks[7]
    epochs = opt.epochs
    lambda1 = opt.lambda1
    batch_size = opt.batch_size
    expname = opt.expname
    append = opt.append
    text_file = open(expname + "_trainloss.txt", "w")
    text_file.close()
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L2Loss()
    metric = mx.metric.CustomMetric(facc)
    metricl = mx.metric.CustomMetric(facc)
    metric2 = mx.metric.MSE()
    loss_rec_G2 =[]
    acc2_rec = []
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    loss_rec_D2 = []
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
            fake_latent= netEn(real_in)
            real_latent = nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx)
            fake_out = netDe(fake_latent)
            fake_concat =  nd.concat(real_in, fake_out, dim=1) if append else  fake_out
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history imagesi
                output = netD(fake_concat)
                output2 = netD2(fake_latent)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                fake_latent_label = nd.zeros(output2.shape, ctx=ctx)
                eps = nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx)
                rec_output = netD(netDe(eps))
                errD_fake = GAN_loss(rec_output, fake_label)
                errD_fake2 = GAN_loss(output, fake_label)
                errD2_fake = GAN_loss(output2, fake_latent_label)
                metric.update([fake_label, ], [output, ])
                metricl.update([fake_latent_label, ], [output2, ])
                real_concat =  nd.concat(real_in, real_out, dim=1) if append else real_out
                output = netD(real_concat)
                output2 = netD2(real_latent)
                real_label = nd.ones(output.shape, ctx=ctx)
                real_latent_label =  nd.ones(output2.shape, ctx=ctx)
                errD_real = GAN_loss(output, real_label)
                errD2_real =  GAN_loss(output2, real_latent_label)
                #errD = (errD_real + 0.5*(errD_fake+errD_fake2)) * 0.5
                errD = (errD_real +(errD_fake)) * 0.5
                errD2 = (errD2_real + errD2_fake) * 0.5
                errD.backward()
                errD2.backward()
                metric.update([real_label, ], [output, ])
                metricl.update([real_latent_label, ], [output2, ])
            trainerD.step(batch.data[0].shape[0])
            trainerD2.step(batch.data[0].shape[0])
            ############################
            # (2) Update G network: maximize log(D(x, G(x, z))) - lambda1 * L1(y, G(x, z))
            ###########################
            with autograd.record():
                eps = nd.random_normal(loc=0, scale=1, shape=fake_latent.shape, ctx=ctx)
                rec_output = netD(netDe(eps))
                fake_latent= (netEn(real_in))
                output2 = netD2(fake_latent)
                fake_out = netDe(fake_latent)
                fake_concat = nd.concat(real_in, fake_out, dim=1) if append else fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                real_latent_label = nd.ones(output2.shape, ctx=ctx)
                errG2 = GAN_loss(rec_output, real_label)
                errG = errG2 + GAN_loss(output2, real_latent_label) + L1_loss(real_out, fake_out) * lambda1
                errR = L1_loss(real_out, fake_out)
                errG.backward()
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
        loss_rec_G2.append(nd.mean(errG2).asscalar())
        loss_rec_G.append(nd.mean(errG-errG2).asscalar()-nd.mean(errR).asscalar()*lambda1)
        loss_rec_D.append(nd.mean(errD).asscalar())
        loss_rec_R.append(nd.mean(errR).asscalar())
        loss_rec_D2.append(nd.mean(errD2).asscalar())

        name, acc = metric.get()
        _, acc2 = metricl.get()
        acc2_rec.append(acc2)
        acc_rec.append(acc)
        # Print log infomation every ten batches
        if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info('discriminator loss = %f, D2 loss = %f, generator loss = %f, binary training acc = %f , '
                             'D2 acc = %f,  reconstruction error= %f at iter %d epoch %d'
                            % (nd.mean(errD).asscalar(), nd.mean(errD2).asscalar(),
                            nd.mean(errG).asscalar(), acc, acc2, nd.mean(errR).asscalar(), iter, epoch))
        iter = iter + 1
        btic = time.time()

        name, acc = metric.get()
        _, acc2 = metricl.get()
        text_tl = open(expname + "_trainloss.txt", "a")
        text_tl.write('%f %f %f %f %f %f %f ' % (nd.mean(errD).asscalar(), nd.mean(errD2).asscalar(),
                    nd.mean(errG).asscalar(), acc, acc2, nd.mean(errR).asscalar(), epoch))
        text_file.close()
        metricl.reset()
        metric.reset()
        train_data.reset()

        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D2.params"
            netD2.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_params(filename)
            fake_img1 = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2 = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3 = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            val_data.reset()
            text_file = open(expname + "_validtest.txt", "a")
            for vbatch in val_data:
                real_in = vbatch.data[0].as_in_context(ctx)
                real_out = vbatch.data[1].as_in_context(ctx)

                fake_latent= netEn(real_in)
                y = netDe(fake_latent)
                fake_out = y
                metric2.update([fake_out, ], [real_out, ])
                _, acc2 = metric2.get()
            text_file.write("%s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2)))
            metric2.reset()
            fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img = nd.concat(fake_img1,fake_img2, fake_img3,fake_img1T,fake_img2T, fake_img3T,dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
            text_file.close()
    return([loss_rec_D,loss_rec_G, loss_rec_R, acc_rec, loss_rec_D2])




def trainAE(opt, train_data, val_data, ctx, networks):

    netEn = networks[0]
    netDe = networks[1]
    trainerEn = networks[4]
    trainerDe = networks[5]
    epochs = opt.epochs
    batch_size = opt.batch_size
    expname = opt.expname
    text_file = open(expname + "_trainloss.txt", "w")
    text_file.close()
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    L1_loss = gluon.loss.L2Loss()
    metric2 = mx.metric.MSE()
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    loss_rec_D2 = []
    stamp = datetime.now().strftime('%Y_%m_%d-%H_%M')
    logging.basicConfig(level=logging.DEBUG)
    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        train_data.reset()
        iter = 0
        for batch in train_data:
            real_in = batch.data[0].as_in_context(ctx)
            real_out = batch.data[1].as_in_context(ctx)
            with autograd.record():
                fake_out = netDe(netEn(real_in))
                errR = L1_loss(real_out, fake_out)
                errR.backward()
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
        loss_rec_R.append(nd.mean(errR).asscalar())

        if iter % 10 == 0:
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info('reconstruction error= %f at iter %d epoch %d'
                            % (nd.mean(errR).asscalar(), iter, epoch))
        iter = iter + 1
        btic = time.time()
        text_tl = open(expname + "_trainloss.txt", "a")
        text_tl.write('%f %f %f %f %f %f %f ' % (0, 0, 0, 0, 0, nd.mean(errR).asscalar(), epoch))
        text_file.close()
        train_data.reset()
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_params(filename)
            fake_img1 = nd.concat(real_in[0],real_out[0], output[0], dim=1)
            fake_img2 = nd.concat(real_in[1],real_out[1], output[1], dim=1)
            fake_img3 = nd.concat(real_in[2],real_out[2], output[2], dim=1)
            val_data.reset()
            text_file = open(expname + "_validtest.txt", "a")
            for vbatch in val_data:
                real_in = vbatch.data[0].as_in_context(ctx)
                real_out = vbatch.data[1].as_in_context(ctx)
                fake_out = netDe(netDe(real_in))
                metric2.update([fake_out, ], [real_out, ])
                _, acc2 = metric2.get()
            text_file.write("%s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2)))
            metric2.reset()
            fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img = nd.concat(fake_img1, fake_img2, fake_img3, fake_img1T, fake_img2T, fake_img3T, dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
            text_file.close()
    return([loss_rec_D,loss_rec_G, loss_rec_R, acc_rec, loss_rec_D2])


def traincvpr18(opt, train_data, val_data, ctx, networks):

    netEn = networks[0]
    netDe = networks[1]
    netD = networks[2]
    trainerEn = networks[4]
    trainerDe = networks [5]
    trainerD =networks[6]
    epochs = opt.epochs
    lambda1 = opt.lambda1
    batch_size = opt.batch_size
    expname = opt.expname
    append = opt.append
    text_file = open(expname + "_trainloss.txt", "w")
    text_file.close()
    text_file = open(expname + "_validtest.txt", "w")
    text_file.close()
    GAN_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
    L1_loss = gluon.loss.L2Loss()
    metric = mx.metric.CustomMetric(facc)
    metricl = mx.metric.CustomMetric(facc)
    metric2 = mx.metric.MSE()
    loss_rec_G2 =[]
    acc2_rec = []
    loss_rec_G = []
    loss_rec_D = []
    loss_rec_R = []
    acc_rec = []
    loss_rec_D2 = []
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
            fake_latent = netEn(real_in)
            fake_out = netDe(fake_latent)
            fake_concat = nd.concat(real_in, fake_out, dim=1) if append else fake_out
            with autograd.record():
                # Train with fake image
                # Use image pooling to utilize history imagesi
                output = netD(fake_concat)
                fake_label = nd.zeros(output.shape, ctx=ctx)
                errD_fake = GAN_loss(output, fake_label)
                metric.update([fake_label, ], [output, ])
                real_concat = nd.concat(real_in, real_out, dim=1) if append else real_out
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
                fake_latent = (netEn(real_in))
                fake_out = netDe(fake_latent)
                fake_concat = nd.concat(real_in, fake_out, dim=1) if append else fake_out
                output = netD(fake_concat)
                real_label = nd.ones(output.shape, ctx=ctx)
                errG = errG2 + GAN_loss(output, real_label) + L1_loss(real_out, fake_out) * lambda1
                errR = L1_loss(real_out, fake_out)
                errG.backward()
            trainerDe.step(batch.data[0].shape[0])
            trainerEn.step(batch.data[0].shape[0])
        loss_rec_G.append(nd.mean(errG-errG2).asscalar()-nd.mean(errR).asscalar()*lambda1)
        loss_rec_D.append(nd.mean(errD).asscalar())
        loss_rec_R.append(nd.mean(errR).asscalar())
        name, acc = metric.get()
        acc_rec.append(acc)
        # Print log infomation every ten batches
        if iter % 10 == 0:
                name, acc = metric.get()
                logging.info('speed: {} samples/s'.format(batch_size / (time.time() - btic)))
                logging.info('discriminator loss = %f, generator loss = %f, binary training acc = %f , reconstruction error= %f at iter %d epoch %d'
                            % (nd.mean(errD).asscalar(), nd.mean(errG).asscalar(), acc, nd.mean(errR).asscalar(), iter, epoch))
        iter = iter + 1
        btic = time.time()

        name, acc = metric.get()
        _, acc2 = metricl.get()
        text_tl = open(expname + "_trainloss.txt", "a")
        text_tl.write('%f %f %f %f %f %f %f ' % (nd.mean(errD).asscalar(), 0,
                    nd.mean(errG).asscalar(), acc, 0, nd.mean(errR).asscalar(), epoch))
        text_file.close()
        metricl.reset()
        metric.reset()
        train_data.reset()

        logging.info('\nbinary training acc at epoch %d: %s=%f' % (epoch, name, acc))
        logging.info('time: %f' % (time.time() - tic))
        if epoch%10 ==0:
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_D.params"
            netD.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_En.params"
            netEn.save_params(filename)
            filename = "checkpoints/"+expname+"_"+str(epoch)+"_De.params"
            netDe.save_params(filename)
            fake_img1 = nd.concat(real_in[0], eal_out[0], fake_out[0], dim=1)
            fake_img2 = nd.concat(real_in[1], real_out[1], fake_out[1], dim=1)
            fake_img3 = nd.concat(real_in[2], real_out[2], fake_out[2], dim=1)
            val_data.reset()
            text_file = open(expname + "_validtest.txt", "a")
            for vbatch in val_data:
                real_in = vbatch.data[0].as_in_context(ctx)
                real_out = vbatch.data[1].as_in_context(ctx)
                fake_latent= netEn(real_in)
                y = netDe(fake_latent)
                fake_out = y
                metric2.update([fake_out, ], [real_out, ])
                _, acc2 = metric2.get()
            text_file.write("%s %s %s\n" % (str(epoch), nd.mean(errR).asscalar(), str(acc2)))
            metric2.reset()
            fake_img1T = nd.concat(real_in[0],real_out[0], fake_out[0], dim=1)
            fake_img2T = nd.concat(real_in[1],real_out[1], fake_out[1], dim=1)
            fake_img3T = nd.concat(real_in[2],real_out[2], fake_out[2], dim=1)
            fake_img = nd.concat(fake_img1,fake_img2, fake_img3, fake_img1T, fake_img2T, fake_img3T, dim=2)
            visual.visualize(fake_img)
            plt.savefig('outputs/'+expname+'_'+str(epoch)+'.png')
            text_file.close()
    return([loss_rec_D,loss_rec_G, loss_rec_R, acc_rec, loss_rec_D2])

