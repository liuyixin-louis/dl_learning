import argparse
import datetime
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from lib.checkpoint import CheckPoint
from .option import Option

import utils as utils
from lib.dataloader import *
from lib.model_builder import get_model
from models.preresnet import PreBasicBlock
from models.resnet import BasicBlock, Bottleneck
from utils.logger import get_logger
from utils.tensorboard_logger import TensorboardLogger
from utils.write_log import write_settings
from lib.trainer import Trainer



class Experiment(object):
    """
    Run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.model = None
        self.epoch = 0
        self.trainer = None

        os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.gpu

        self.settings.set_save_path()
        write_settings(self.settings)
        self.logger = get_logger(self.settings.save_path, "dcp")
        self.tensorboard_logger = TensorboardLogger(self.settings.save_path)
        self.settings.copy_code(self.logger, src=os.path.abspath('./'),
                                dst=os.path.join(self.settings.save_path, 'code'))
        self.logger.info("|===>Result will be saved at {}".format(self.settings.save_path))

        self.prepare()


        # put your job-ralated varible below

    def prepare(self):
        """
        Preparing experiments
        """

        self._set_gpu()
        self._set_dataloader()
        self._set_model()
        self._set_checkpoint()
        self._set_trainier()
        torch.set_num_threads(4)

    def _set_gpu(self):
        """
        Initialize the seed of random number generator
        """

        # set torch seed
        # init random seed
        torch.manual_seed(self.settings.seed)
        torch.cuda.manual_seed(self.settings.seed)
        torch.cuda.set_device(0)
        cudnn.benchmark = True

    def _set_dataloader(self):
        """
        Create train loader and validation loader for channel pruning
        """

        if 'cifar' in self.settings.dataset:
            self.train_loader, self.val_loader = get_cifar_dataloader(self.settings.dataset,
                                                                      self.settings.batch_size,
                                                                      self.settings.n_threads,
                                                                      self.settings.data_path,
                                                                      self.logger)
        elif self.settings.dataset in ['imagenet']:
            self.train_loader, self.val_loader = get_imagenet_dataloader(self.settings.dataset,
                                                                         self.settings.batch_size,
                                                                         self.settings.n_threads,
                                                                         self.settings.data_path,
                                                                         self.logger)
        elif self.settings.dataset in ['sub_imagenet']:
            num_samples_per_category = self.settings.max_samples // 1000
            self.train_loader, self.val_loader = get_sub_imagenet_dataloader(self.settings.dataset,
                                                                             self.settings.batch_size,
                                                                             num_samples_per_category,
                                                                             self.settings.n_threads,
                                                                             self.settings.data_path,
                                                                             self.logger)

    def _set_trainier(self):
        """
        trainer
        """
        self.trainer = Trainer(self.model, self.train_loader, self.val_loader, self.settings, self.logger,
                 self.tensorboard_logger, optimizer_state=None, run_count=0))



    def _set_model(self):
        """
        Available model
        cifar:
            preresnet
        imagenet:
            resnet
        """

        self.model, self.test_input = get_model(self.settings.dataset,
                                                         self.settings.net_type,
                                                         self.settings.depth,
                                                         self.settings.n_classes)

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.model is not None and self.pruned_model is not None, "please create model first"

        self.checkpoint = CheckPoint(self.settings.save_path, self.logger)
        self._load_pretrained()
        self._load_resume()

    def _load_pretrained(self):
        """
        Load pre-trained model
        """

        if self.settings.pretrained is not None:
            check_point_params = torch.load(self.settings.pretrained)
            model_state = check_point_params['model']
            self.model = self.checkpoint.load_state(self.model, model_state)
            self.logger.info("|===>load restrain file: {}".format(self.settings.pretrained))

    def _load_resume(self):
        """
        Load resume checkpoint
        """

        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            model_state = check_point_params["model"]
            
            self.model = self.checkpoint.load_state(self.model, model_state)
            
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))
    
    def run(self):
        """put you job here"""
        """
        Learn the parameters of the additional classifier and
        fine tune model with the additional losses and the final loss
        """

        best_top1 = 100
        best_top5 = 100
        start_epoch = 0

        # if load resume checkpoint
        if self.epoch != 0:
            start_epoch = self.epoch + 1
            self.epoch = 0

        # self.trainer.val(0)

        for epoch in range(start_epoch, self.settings.n_epochs):
            train_error, train_loss, train5_error = self.trainer.train(epoch)
            val_error, val_loss, val5_error = self.trainer.val(epoch)

            # write log
            log_str = "{:d}\t".format(epoch)
            for i in range(len(train_error)):
                log_str += "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
                    train_error[i], train_loss[i], val_error[i],
                    val_loss[i], train5_error[i], val5_error[i])
            write_log(self.settings.save_path, 'log.txt', log_str)

            # save model and checkpoint
            best_flag = False
            if best_top1 >= val_error[-1]:
                best_top1 = val_error[-1]
                best_top5 = val5_error[-1]
                best_flag = True

            if best_flag:
                self.checkpoint.save_checkpoint(self.trainer.model,self.trainer.optimizer,self.epoch)

            self.logger.info("|===>Best Result is: Top1 Error: {:f}, Top5 Error: {:f}\n".format(best_top1, best_top5))
            self.logger.info("|==>Best Result is: Top1 Accuracy: {:f}, Top5 Accuracy: {:f}\n".format(100 - best_top1,
                                                                                                     100 - best_top5))

    


def main():
    parser = argparse.ArgumentParser(description="Experiments")
    
    parser.add_argument('--conf_path', type=str, metavar='conf_path',
                        help='configuration path')
    parser.add_argument('--model_path', type=str, metavar='model_path',
                        help='model path of the pretrained model')

    args = parser.parse_args()

    option = Option(args.conf_path)
    if args.model_path:
        option.pretrained = args.model_path

    experiment = Experiment(option)

    # your job
    experiment.run()



if __name__ == '__main__':
    main()
