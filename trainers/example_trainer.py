# coding=utf-8
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from trainers.base_trainer import BaseTrainer
from utils import utils

class ExampleTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, config, logger):
        super(ExampleTrainer, self).__init__(model, train_loader, val_loader, config, logger)
        self.optimizerG = self.create_optimization(self.model.netG, self.config["learning_rate_g"])
        self.optimizerD = self.create_optimization(self.model.netD, self.config["learning_rate_d"])


    def train(self):
        total_epoch_num = self.config['num_epochs']
        for cur_epoch in range(1, total_epoch_num+1):
            self.cur_epoch = cur_epoch
            self.train_epoch()
            # self.evaluate_epoch()   # uncomment if neccessary

            # printe information of an epoch
            self.logger.log_printer.epoch_case_print()

            # save model
            self.model.save(self.model.netG)
            self.model.save(self.model.netD)

            # logger
            self.logger.write_info_to_logger(variable_dict={})
            self.logger.write()


    def train_epoch(self):
        """
        training in a epoch
        :return: 
        """
        # Learning rate adjustment
        self.learning_rateG = self.adjust_learning_rate(self.optimizerG, self.cur_epoch,
                                                        self.config["learning_rate_g"],
                                                        self.config["learning_rate_decay_g"],
                                                        self.config["learning_rate_decay_epoch_g"])
        self.learning_rateD = self.adjust_learning_rate(self.optimizerD, self.cur_epoch,
                                                        self.config["learning_rate_d"],
                                                        self.config["learning_rate_decay_d"],
                                                        self.config["learning_rate_decay_epoch_d"])
        self.train_lossesG = utils.AverageMeter()
        self.train_lossesD = utils.AverageMeter()

        for batch_idx, item in enumerate(tqdm(self.train_loader)):
            self.batch_idx = batch_idx + 1
            batch_x, batch_y = item
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(async=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.train_step(batch_x_var, batch_y_var)

            # print your loss with self.train_lossesG.avg, self.train_lossesD.avg
            self.logger.log_printer.iter_case_print()



    def train_step(self, images, GTmatte):
        """
        training in a step
        :param images: 
        :param labels: 
        :return: 
        """
        # train D
        if self.batch_idx % self.config["every_d"]:
            # Set the model to be in training mode (for dropout and batchnorm)
            self.model.netG.eval()
            self.model.netD.train()

            # real
            discr_real_d = self.model.netD(images)
            target_real_label = t.tensor(1.0)
            target_real = target_real_label.expand_as(discr_real_d)
            if torch.cuda.is_available():
                target_real = target_real.cuda()
            loss_real_d = self.get_lossD(discr_real_d, target_real)
            # fake
            pred_fake_d = self.model.netG(images)
            target_fake_label = t.tensor(0.0)
            target_fake = target_fake_label.expand_as(pred_fake_d)
            if torch.cuda.is_available():
                target_fake = target_fake.cuda()
            loss_fake_d = self.get_lossD(pred_fake_d, target_fake)

            # loss sum
            self.loss_D = loss_real_d + loss_fake_d
            self.train_lossesD.update(self.loss_D, images.size(0))

            # Optimization step
            if torch.cuda.device_count() > 1 and torch.cuda.is_available():
                self.optimizerD.module.zero_grad()
            else:
                self.optimizerD.zero_grad()
            self.loss_D.backward()
            if torch.cuda.device_count() > 1 and torch.cuda.is_available():
                self.optimizerD.module.step()
            else:
                self.optimizerD.step()


        # train G
        if self.batch_idx % self.config["every_g"]:
            # Set the model to be in training mode (for dropout and batchnorm)
            self.model.netG.train()
            self.model.netD.eval()

            pred_fake_g = self.model.netG(images)
            loss_pred_g = self.get_lossG(pred_fake_g, GTmatte)
            discr_fake_d = self.model.netD(pred_fake_g)
            target_fake = t.tensor(1.0).expand_as(discr_fake_d)
            if torch.cuda.is_available():
                target_fake = target_fake.cuda()
            loss_discr_d = self.get_lossD(discr_fake_d, target_fake)

            # loss sum
            self.loss_G = loss_pred_g + loss_discr_d
            self.train_lossesG.update(self.loss_G, images.size(0))

            # Optimization step
            if torch.cuda.device_count() > 1 and torch.cuda.is_available():
                self.optimizerG.module.zero_grad()
            else:
                self.optimizerG.zero_grad()
            self.loss_G.backward()
            if torch.cuda.device_count() > 1 and torch.cuda.is_available():
                self.optimizerG.module.step()
            else:
                self.optimizerG.step()


    def get_lossG(self, gener_imgs, org_imgs):
        criterion = nn.SmoothL1Loss()
        if torch.cuda.is_available():
            criterion.cuda()
        return criterion(gener_imgs, org_imgs)

    def get_lossD(self, discrs, targets):
        criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            criterion.cuda()
        return criterion(discrs, targets)


    def create_optimization(self, net, learning_rate):
        """
        optimizer
        :return:
        """
        optimizer = torch.optim.Adam(net.parameters(),
                                          lr=learning_rate, weight_decay=0) #lr:1e-4
        if torch.cuda.device_count() > 1:
            print('optimizer device_count: ',torch.cuda.device_count())
            optimizer = nn.DataParallel(optimizer,device_ids=range(torch.cuda.device_count()))
        """
        # optimizing parameters seperately
        ignored_params = list(map(id, net.fc.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                            net.parameters())
        optimizer = torch.optim.Adam([
            {'params': base_params},
            {'params': net.fc.parameters(), 'lr': 1e-3}
            ], lr=1e-2, betas=(0.9, 0.99), eps=1e-08, weight_decay=0, amsgrad=False)"""
        return optimizer


    def adjust_learning_rate(self, optimizer, epoch, learning_rate, learning_rate_decay, learning_rate_decay_epoch):
        """
        decay learning rate
        :param optimizer: 
        :param epoch: the first epoch is 1
        :return: 
        """
        # """Decay Learning rate at 1/2 and 3/4 of the num_epochs"""
        # lr = lr_init
        # if epoch >= num_epochs * 0.75:
        #     lr *= decay_rate ** 2
        # elif epoch >= num_epochs * 0.5:
        #     lr *= decay_rate
        learning_rate = learning_rate * (learning_rate_decay ** ((epoch - 1) // learning_rate_decay_epoch))
        if torch.cuda.device_count() > 1 and torch.cuda.is_available():
            for param_group in optimizer.module.param_groups:
                param_group['lr'] = learning_rate
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

        return learning_rate


    def evaluate_epoch(self):
        """
        evaluating in a epoch in an visualization way or some
        :return:
        """
        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.netG.eval()
        self.model.netD.eval()
        for batch_idx, item in enumerate(self.val_loader):
            batch_x, batch_y = item
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(async=self.config['async_loading']), batch_y.cuda(
                    async=self.config['async_loading'])
            batch_x_var, batch_y_var = Variable(batch_x), Variable(batch_y)
            self.evaluate_step(batch_x_var, batch_y_var)


    def evaluate_step(self, images, GTmatte):
        """
        evaluating in a step
        :param images:
        :param labels:
        :return:
        """
        with torch.no_grad():
            infer = self.model.netG(images)
            # implement the logic of calculating loss like train phase or visualizing the generative images here
            pass
        raise NotImplementedError("Please implement the logic of visualizing the generative images here")