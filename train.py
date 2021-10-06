from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
import os
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from models import *
from utils.log import Log
from torch.utils.tensorboard import SummaryWriter


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # logger
        self.logger = Log(self.args.log_dir, self.args.model_name).get_logger()
        self.logger.info(json.dumps(vars(self.args)))

        # state
        self.state_path = os.path.join(self.args.state_dir, self.args.model_name)
        if not os.path.exists(self.state_path):
            os.makedirs(self.state_path)

        # warm up
        # self.gradual_warmup_steps = [i * self.args.lr for i in torch.linspace(0.5, 2.0, 7)]
        # self.lr_decay_epochs = range(14, 47, self.args.lr_decay_step)

        # early stop
        self.early_stop = 0

        # data loader
        self.logger.info("starting process data!")
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = torchvision.datasets.CIFAR10(root=self.args.data_path, train=True, download=False, transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(root=self.args.data_path, train=False, download=False, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False, num_workers=2)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.logger.info("finish process data!")

        # model
        if self.args.model_name == "my_model":
            self.net = Net().to(self.args.gpu)
        elif self.args.model_name == "vgg16":
            self.net = VGG('VGG16').to(self.args.gpu)
        elif self.args.model_name == "Resnet18":
            self.net = ResNet18().to(self.args.gpu)
        elif self.args.model_name == "Resnet34":
            self.net = ResNet34().to(self.args.gpu)
        elif self.args.model_name == "Resnet50":
            self.net = ResNet50().to(self.args.gpu)
        elif self.args.model_name == "Resnet101":
            self.net = ResNet101().to(self.args.gpu)
        elif self.args.model_name == "GoogLeNet":
            self.net = GoogLeNet().to(self.args.gpu)
        elif self.args.model_name == "DenseNet121":
            self.net = DenseNet121().to(self.args.gpu)
        elif self.args.model_name == "MobileNet":
            self.net = MobileNet().to(self.args.gpu)
        elif self.args.model_name == "MobileNetv2":
            self.net = MobileNetV2().to(self.args.gpu)

        # loss
        self.criterion = nn.CrossEntropyLoss()

        # optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)
        if self.args.optimizer == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        if self.args.optimizer == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr)

    def save_checkpoint(self, step):
        state = self.net.state_dict()
        # delete previous checkpoint
        for filename in os.listdir(self.state_path):
            if self.args.model_name in filename.split('.') and os.path.isfile(os.path.join(self.state_path, filename)):
                os.remove(os.path.join(self.state_path, filename))
        # save checkpoint
        torch.save(state, os.path.join(self.args.state_dir, self.args.model_name,
                                       self.args.model_name + '.' + str(step) + '.ckpt'))

    def save_model(self, best_step):
        os.rename(os.path.join(self.state_path, self.args.model_name + '.' + str(best_step) + '.ckpt'),
                  os.path.join(self.state_path, self.args.model_name + '.best'))

    def before_test_load(self):
        state = torch.load(os.path.join(self.state_path, self.args.model_name + '.best'), map_location=self.args.gpu)
        self.net.load_state_dict(state)

    # 要打印的信息较少 暂时不用到此函数
    def print_per_batch(self, prefix, metrics):
        self.logger.info(' ')
        self.logger.info('------------------------------------')
        self.logger.info('this batch')
        if prefix == 'train':
            self.logger.info('loss: {:.4f}'.format(np.mean(metrics.losses)))
        self.logger.info('macro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.macro_f1,
                                                                                      metrics.macro_recall,
                                                                                      metrics.macro_precision))
        self.logger.info('micro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.micro_f1,
                                                                                      metrics.micro_recall,
                                                                                      metrics.micro_precision))
        self.logger.info("------------------------------------")

    # 要打印的信息较少 暂时不用到此函数
    def print_per_epoch(self, prefix, metrics, epoch):
        self.logger.info('------------------------------------')
        if prefix == 'train':
            self.logger.info('epoch: {} | loss: {:.4f}'.format(epoch, np.mean(self.train_metrics.losses)))
        else:
            self.logger.info('final test results:')
        if prefix == 'train':
            self.logger.info('macro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.macro_f1,
                                                                                          metrics.macro_recall,
                                                                                          metrics.macro_precision))
            self.logger.info('micro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.micro_f1,
                                                                                          metrics.micro_recall,
                                                                                          metrics.micro_precision))
        else:
            self.logger.info('macro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.epoch_macro_f1,
                                                                                          metrics.epoch_macro_recall,
                                                                                          metrics.epoch_macro_precision))
            self.logger.info('micro: f1:{:.4f}\t recall:{:.4f}\t precision:{:.4f}'.format(metrics.epoch_micro_f1,
                                                                                          metrics.epoch_micro_recall,
                                                                                          metrics.epoch_micro_precision))
        self.logger.info("------------------------------------")

    def train(self):
        self.logger.info('start training')
        writer = SummaryWriter()
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(self.args.num_epoch):

            running_loss = 0.
            total = 0
            correct = 0
            train_acc = 0.
            # net.train()
            prefix = "train"
            tq = tqdm(self.trainloader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
            for data in tq:
                inputs, labels = data
                inputs, labels = inputs.to(self.args.gpu), labels.to(self.args.gpu)

                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                train_acc = correct / total

                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                self.logger.info('epoch:%d, loss: %.4f, acc:%.4f' % (epoch + 1, running_loss/total, train_acc))

            # using visual-tools tensorboard to draw Training process chart
            writer.add_scalar("loss/train/" + self.args.model_name, running_loss/total, global_step=epoch)
            writer.add_scalar("acc/train/" + self.args.model_name, train_acc, global_step=epoch)

            # 保存epoch训练过程中，acc最高的那个epoch的模型
            if train_acc > best_acc:
                best_epoch = epoch
                best_acc = train_acc
                self.save_checkpoint(epoch)
                self.logger.info('best_acc: {:.5f}'.format(best_acc))

        self.logger.info('Finished Training')

        # 保存最优模型
        self.save_model(best_epoch)
        self.logger.info("finish training!")

        # 载入模型
        self.before_test_load()
        self.evaluate(istest=True)

    def evaluate(self, istest=False):
        correct = 0
        total = 0
        # net.eval()
        prefix = "test"
        tq = tqdm(self.testloader, desc='{}:'.format(prefix), ncols=0)
        with torch.no_grad():
            for data in tq:
                images, labels = data
                images, labels = images.to(self.args.gpu), labels.to(self.args.gpu)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        self.logger.info('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in tq:
                images, labels = data
                images, labels = images.to(self.args.gpu), labels.to(self.args.gpu)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            self.logger.info('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))
        self.logger.info('testing finish!')