"""
Trainer
"""
import os
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tet.tet import TET_loss
from util import accuracy, AverageMeter, save_checkpoint, convert_secs2time, print_table

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class STrainer(object):
    """
    Pytorch Trainer for Spiking Neural Network Training
    """
    def __init__(self, model:nn.Module, trainloader, testloader, train_sampler, test_sampler, args, logger) -> None:
        self.args = args

        # model
        self.model = model

        # data loaders
        self.trainloader = trainloader
        self.testloader = testloader
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler

        # distributed trainer
        self.local_rank=int(os.environ['LOCAL_RANK'])
        #self.local_rank = args.local_rank
        self.nprocs = args.nprocs
        torch.cuda.set_device(self.local_rank)

        seed_all(args.seed)
        cudnn.deterministic = True

        # cuda
        self.model = self.model.cuda(self.local_rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank])

        # criterion
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.local_rank)

        # optimizer
        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.args.wd)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        
        # lr scheduler
        if self.args.lr_sch == "step":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.schedule, last_epoch=-1)
        elif self.args.lr_sch == "cos":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=0, T_max=args.epochs)
        
        cudnn.benchmark = True

        # logger
        self.logger = logger
        self.logger_dict = {}
        self.initialize_meters()


    def initialize_meters(self):
        self.tr_loss = AverageMeter()
        self.tr_acc1 = AverageMeter()
        self.tr_acc5 = AverageMeter()

        self.val_loss = AverageMeter()
        self.val_acc1 = AverageMeter()
        self.val_acc5 = AverageMeter()

        self.epoch_time = AverageMeter()

        self.best_acc = 0

    def train_epoch(self):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        # switch to train mode
        self.model.train()

        end = time.time()
        for i, (images, target) in enumerate(self.trainloader):
            # measure data loading time
            #import pdb;pdb.set_trace()
            data_time.update(time.time() - end)

            images = images.cuda(self.local_rank, non_blocking=True)
            target = target.cuda(self.local_rank, non_blocking=True)
            images = images.float()

            # compute output
            output = self.model(images)
            mean_out = torch.mean(output, dim=1)

            loss = TET_loss(output, target, self.criterion, self.args.means, self.args.lamb)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

            torch.distributed.barrier()

            reduced_loss = reduce_mean(loss, self.nprocs)
            reduced_acc1 = reduce_mean(acc1, self.nprocs)
            reduced_acc5 = reduce_mean(acc5, self.nprocs)

            losses.update(reduced_loss.item(), images.size(0))
            top1.update(reduced_acc1.item(), images.size(0))
            top5.update(reduced_acc5.item(), images.size(0))

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        

        # logging
        self.logger_dict["train_loss"] = losses.avg
        self.logger_dict["train_top1"] = top1.avg
        
            
    def valid_epoch(self):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.testloader):
                images = images.cuda(self.local_rank, non_blocking=True)
                target = target.cuda(self.local_rank, non_blocking=True)
                images = images.float()

                # compute output
                output = self.model(images)
                mean_out = torch.mean(output, dim=1)
                loss = self.criterion(mean_out, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(mean_out, target, topk=(1, 5))

                torch.distributed.barrier()

                reduced_loss = reduce_mean(loss, self.nprocs)
                reduced_acc1 = reduce_mean(acc1, self.nprocs)
                reduced_acc5 = reduce_mean(acc5, self.nprocs)

                losses.update(reduced_loss.item(), images.size(0))
                top1.update(reduced_acc1.item(), images.size(0))
                top5.update(reduced_acc5.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        self.logger_dict["valid_loss"] = losses.avg
        self.logger_dict["valid_top1"] = top1.avg


    def fit(self):
        self.logger.info("Training Start!\n Model={}\n Optimizer={}".format(self.args.model, self.args.optimizer))

        start_time = time.time()
        for epoch in range(self.args.epochs):
            self.train_sampler.set_epoch(epoch)
            self.test_sampler.set_epoch(epoch)

            self.logger_dict["ep"] = epoch+1
            self.logger_dict["lr"] = self.optimizer.param_groups[0]['lr']
            
            self.train_epoch()

            # lr step
            self.lr_scheduler.step()
            self.valid_epoch()
        
            is_best = self.logger_dict["valid_top1"] > self.best_acc
            if is_best:
                best_acc = self.logger_dict["valid_top1"]

            state = {
                'state_dict': self.model.module.state_dict(),
                'acc': best_acc,
                'epoch': epoch,
                'optimizer': self.optimizer.state_dict(),
            }

            filename=f"checkpoint.pth.tar"
            save_checkpoint(state, is_best, self.args.save_path, filename=filename)

            # terminal log
            columns = list(self.logger_dict.keys())
            values = list(self.logger_dict.values())
            print_table(values, columns, epoch, self.logger)

            # record time
            e_time = time.time() - start_time
            self.epoch_time.update(e_time)
            start_time = time.time()
            
            need_hour, need_mins, need_secs = convert_secs2time(
            self.epoch_time.avg * (self.args.epochs - epoch))
            print('[Need: {:02d}:{:02d}:{:02d}]'.format(
                need_hour, need_mins, need_secs))