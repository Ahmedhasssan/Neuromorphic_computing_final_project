import os
import torch
import numpy as np
import random
import argparse
import logging
from data import build_mnist
from vgg9 import svgg9_mini
from trainers.trainer import STrainer

parser = argparse.ArgumentParser(description='PyTorch SNN Training')
parser.add_argument('--model', type=str, help='model architecture')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--T', default=2, type=int, metavar='N', help='snn simulation time (default: 2)')
parser.add_argument('--means', default=1.0, type=float, metavar='N', help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--lamb', default=0.05, type=float, metavar='N', help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('-b', '--batch-size', default=512, type=int, metavar='N', help='batch size')
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
parser.add_argument('--lr_sch', type=str, default='cos', help='learning rate scheduler')

# dataset
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset: CIFAR10 / ImageNet_1k')
parser.add_argument('--data_path', type=str, default='./data/', help='data directory')
parser.add_argument('--train_dir', type=str, default='./data/', help='data directory')
parser.add_argument('--val_dir', type=str, default='./data/', help='data directory')
parser.add_argument('--save_dir', type=str, default='./data/', help='data directory')

# ddp
parser.add_argument('--seed', type=int, default=1000, help='use random seed to make sure all the processes has the same model')
parser.add_argument("--local_rank", type=int, default=0, help="Local rank [required]")

# save path
parser.add_argument('--save_path', type=str, default='./save/', help='Folder to save checkpoints and log.')
parser.add_argument('--log_file', type=str, default=None, help='path to log file')

# Fine-tuning & Evaluate
parser.add_argument('--fine_tune', dest='fine_tune', action='store_true', help='fine tuning from the pre-trained model, force the start epoch be zero')
parser.add_argument('--resume', default='', type=str, help='path of the pretrained model')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

args = parser.parse_args()

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def convert_to_three_channel(image):
    # Duplicate the single channel into three channels
    return torch.cat([image, image, image], dim=0)

# Apply the conversion function to all images in the dataset
def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    # initialize terminal logger
    logger = logging.getLogger('training')
    if args.log_file is not None:
        fileHandler = logging.FileHandler(args.save_path+args.log_file)
        fileHandler.setLevel(0)
        logger.addHandler(fileHandler)
    streamHandler = logging.StreamHandler()
    streamHandler.setLevel(0)
    logger.addHandler(streamHandler)
    logger.root.setLevel(0)
    logger.info(args)
    
    # unify the random seed for all the proesses 
    set_random_seeds(random_seed=args.seed)

    # initialize process
    torch.distributed.init_process_group(backend="nccl")
    args.world_size = torch.distributed.get_world_size()
    
    # data loader
    args.nprocs = 1 if args.evaluate else torch.cuda.device_count()
    args.batch_size = int(args.batch_size / args.nprocs)

    if args.dataset == "MNIST":
        train_dataset, val_dataset = build_mnist(args, use_mnist=False)
        num_classes = 10

    # Training sampler

    #train_dataset.data = torch.stack([convert_to_three_channel(image) for image in train_dataset.data])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            sampler=train_sampler)

    # get model
    if args.model == "vgg9_mini":
        model = svgg9_mini(num_classes=num_classes)
    
    model.T = args.T
    logger.info(model)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    testloader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=args.workers,
                                            pin_memory=True,
                                            sampler=val_sampler)

    # initialize the trainer
    trainer = STrainer(
        model,
        trainloader,
        testloader,
        train_sampler,
        val_sampler,
        args,
        logger
    )

    if args.evaluate:
        cnt = 0
        for m in model.modules():
            if hasattr(m, "layer_idx"):
                m.layer_idx = cnt
                cnt += 1

        trainer.valid_epoch()
        logger.info("Test accuracy = {:.3f}".format(trainer.logger_dict["valid_top1"]))


        exit()

    # start training
    trainer.fit()


if __name__ == '__main__':
    main()