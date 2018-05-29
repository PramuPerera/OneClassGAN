import argparse
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
    parser.add_argument("--seed", default=-1, type=float, help="Seed generator. Use -1 for random.")
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
    parser.add_argument("--noisevar", default=0.02, type=float, help="variance of noise added to input")
    parser.add_argument("--istest", default=1, type=float, help="if test set 1, otherwise validation")
    args = parser.parse_args()
    if args.use_gpu == 1:
        args.use_gpu = True
    else:
        args.use_gpu = False
        
    if args.istest == 1:
        args.istest = True
    else:
        args.istest = False
        
    return args
