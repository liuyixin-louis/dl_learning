import argparse
import datetime
import time

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from channel_selection import LayerChannelSelection
from lib.checkpoint import CheckPoint
from option import Option
from trainer import SegmentWiseTrainer

import dcp.utils as utils
from dcp.pruning import ResModelPrune
from dcp.dataloader import *
from dcp.mask_conv import MaskConv2d
from dcp.model_builder import get_model
from dcp.models.preresnet import PreBasicBlock
from dcp.models.resnet import BasicBlock, Bottleneck
from dcp.utils.logger import get_logger
from dcp.utils.model_analyse import ModelAnalyse
from dcp.utils.tensorboard_logger import TensorboardLogger
from dcp.utils.write_log import write_settings
from dcp.utils import cal_pivot


class Experiment(object):
    """
    Run experiments with pre-defined pipeline
    """

    def __init__(self, options=None, conf_path=None):
        self.settings = options or Option(conf_path)
        self.checkpoint = None
        self.train_loader = None
        self.val_loader = None
        self.original_model = None
        self.epoch = 0

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
        Initialize segment-wise trainer trainer
        """

        # initialize segment-wise trainer
        self.segment_wise_trainer = SegmentWiseTrainer(original_model=self.original_model,
                                                       pruned_model=self.pruned_model,
                                                       train_loader=self.train_loader,
                                                       val_loader=self.val_loader,
                                                       settings=self.settings,
                                                       logger=self.logger,
                                                       tensorboard_logger=self.tensorboard_logger)
        if self.aux_fc_state is not None:
            self.segment_wise_trainer.update_aux_fc(self.aux_fc_state)


    def _set_model(self):
        """
        Available model
        cifar:
            preresnet
        imagenet:
            resnet
        """

        self.original_model, self.test_input = get_model(self.settings.dataset,
                                                         self.settings.net_type,
                                                         self.settings.depth,
                                                         self.settings.n_classes)

    def _set_checkpoint(self):
        """
        Load pre-trained model or resume checkpoint
        """

        assert self.original_model is not None and self.pruned_model is not None, "please create model first"

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
            self.original_model = self.checkpoint.load_state(self.original_model, model_state)
            self.logger.info("|===>load restrain file: {}".format(self.settings.pretrained))

    def _load_resume(self):
        """
        Load resume checkpoint
        """

        if self.settings.resume is not None:
            check_point_params = torch.load(self.settings.resume)
            original_model_state = check_point_params["original_model"]
            
            self.original_model = self.checkpoint.load_state(self.original_model, original_model_state)
            
            self.logger.info("|===>load resume file: {}".format(self.settings.resume))

    


def main():
    parser = argparse.ArgumentParser(description="Experiments")
    parser.add_argument('conf_path', type=str, metavar='conf_path',
                        help='configuration path')
    parser.add_argument('--model_path', type=str, metavar='model_path',
                        help='model path of the pruned model')
    args = parser.parse_args()

    option = Option(args.conf_path)
    if args.model_path:
        option.pretrained = args.model_path
    if args.softmax_weight:
        option.softmax_weight = args.softmax_weight
    if args.mse_weight:
        option.mse_weight = args.mse_weight

    experiment = Experiment(option)

    # your job
    job()

def job():
    pass

if __name__ == '__main__':
    main()
