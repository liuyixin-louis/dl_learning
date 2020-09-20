import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
import logging
import os
import sys
from models.cifar.resnet_cifar import resnet56_cifar,resnet20_cifar,resnet32_cifar
from models.imagenet.resnet import resnet152


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """
    logger = logging.getLogger(logger_name)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    console_formatter = logging.Formatter('%(message)s')

    
    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger


class Trainer(object):
    def __init__(self):
        self.paser()
        self.logger = get_logger('/home/young/liuyixin/dl_learning/material/logs','testLogger')
        self.get_data()
        self.get_model()
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=0.01, rho=0.9, eps=1e-06, weight_decay=0)

    def get_model(self):
        self.args.model_type = self.args.model_type.lower()
        if self.args.datasets == 'cifar10':
            if self.args.model_type == 'resnet56':
                self.model = resnet56_cifar()
            elif self.args.model_type == 'resnet20':
                self.model = resnet20_cifar()
            elif self.args.model_type == 'resnet32':
                self.model = resnet32_cifar()
        elif self.args.datasets == 'mnist':
            if self.args.model_type == "testnet":
                self.model = TestNet()
            elif self.args.model_type == "simplenet_v2":
                self.model = Simplenet_v2()
        elif self.args.datasets == "imagenet":
            if self.args.model_type == 'resnet152':
                self.model = resnet152()
        self.model = self.model.cuda()
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, range(self.args.n_gpu))

    def get_data(self):
        assert self.args
        self.args.datasets = self.args.dataset.lower()
        if self.args.datasets == 'mnist':
            self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./material/dataset/minist/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            self.test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('./material/dataset/minist/', train=False, transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                batch_size=self.args.test_batch_size, shuffle=True, **self.kwargs)
            self.logger.info("|===>Mnist Dataset loading succeed!<=====|")
        elif  self.args.datasets == 'cifar10':
            transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            self.train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('/home/dataset/cifar', train=True, download=True,
                        transform=transform),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
            self.test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('/home/dataset/cifar', train=False, transform=transform),
                batch_size=self.args.test_batch_size, shuffle=True, **self.kwargs)
            self.logger.info("|===>CIFAR10 Dataset loading succeed!<=====|")
        elif self.args.datasets == 'imagenet':
            traindir = os.path.join('/home/dataset/imagenet', 'train')
            valdir = os.path.join('/home/dataset/imagenet', 'val')

            # preprocessing
            input_size = 224
            imagenet_tran_train = [
                transforms.RandomResizedCrop(input_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
            imagenet_tran_test = [
                transforms.Resize(int(input_size / 0.875)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]

            self.train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(traindir, transforms.Compose(imagenet_tran_train)),
                batch_size=self.args.test_batch_size, shuffle=True,sampler=None,
                **self.kwargs )

            self.test_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose(imagenet_tran_test)),
                batch_size=self.args.test_batch_size, shuffle=False,
                **self.kwargs)
            self.logger.info("|===>imagenet Dataset loading succeed!<=====|")
        
        
    def train(self,args, model,device, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = self.criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item())
                print("\r",msg,end="",flush=True)
                # self.logger.info(msg)

    
    def test(self,args, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += self.criterion(output, target).item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        msg = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset))
        print("\r",msg,end="",flush=True)
        # self.logger.info(msg)

    
    def paser(self):
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--model-type',type=str,default='resnet56')
        parser.add_argument('--dataset',type=str,default='cifar10')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                            help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                            help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
                            help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                            help='how many batches to wait before logging training status')
        parser.add_argument('--save-model', action='store_true', default=False,
                            help='For Saving the current Model')
        parser.add_argument('--num-workers', type=int, default=4)
        parser.add_argument('--n_gpu', type=int, default=4)
        self.args = parser.parse_args()
        self.use_cuda = not self.args.no_cuda and torch.cuda.is_available()

        torch.manual_seed(self.args.seed)

        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.kwargs = {'num_workers': self.args.num_workers, 'pin_memory': True} if self.use_cuda else {}
    
    def start(self):
        for epoch in range(1, self.args.epochs + 1):
            self.train(self.args, self.model, self.device, self.train_loader, self.optimizer, epoch)
            self.test(self.args, self.model, self.device, self.test_loader)

        if (self.args.save_model):
            torch.save(self.model.state_dict(),".\material\chekpoint\mnist_cnn.pt")
                


class Simplenet_v2(nn.Module):
    """
    This is Simplenet but with only one small Linear layer, instead of two Linear layers,
    one of which is large.
    26K parameters.
    python compress_classifier.py ${MNIST_PATH} --arch=simplenet_mnist --vs=0 --lr=0.01

    ==> Best [Top1: 98.970   Top5: 99.970   Sparsity:0.00   Params: 26000 on epoch: 54]
    """
    def __init__(self):
        super(Simplenet_v2,self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2, 2) # (20,12,12)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2, 2)# (50,4,4)
        self.avgpool = nn.AvgPool2d(4, stride=1) # (50,1,1)
        self.fc = nn.Linear(50, 10) 
        self.softmax = nn.Softmax(dim=1) #(1,10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))# (20,12,12)
        x = self.pool2(self.relu2(self.conv2(x)))# (50,4,4)
        x = self.avgpool(x)# (50,1,1)
        x = x.view(x.size(0), -1) #(50)
        x = self.fc(x) #(10)

        return x


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet,self).__init__()
        # input (1,28,28)
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # (20,24,24)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, 2) # (20,12,12)
        self.conv2 = nn.Conv2d(20, 50, 5, 1) # (50,8,8)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, 2) # (50,4,4)
        self.fc1 = nn.Linear(4*4*50, 500) #(1,500)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(500, 10) #(1,10)
        self.softmax = nn.Softmax(dim=1) #(1,10)

    def forward(self,x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 4*4*50)
        x = self.relu_fc1(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    trainer = Trainer()
    trainer.start()



if __name__ == '__main__':
    main()
    


