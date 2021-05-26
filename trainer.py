import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import utils


class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError


    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError


    def visualize_step(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError


    def metric_step(self, gt_dict, pred_dict, multiFrame=False):
        ''' Performs a metric step.
        '''
        metrics_dict = {}
        gt_dict = gt_dict.copy()
        pred_dict = pred_dict.copy()

        if multiFrame:
            for _dict in (gt_dict, pred_dict):
                for k,v in _dict.items():
                    shape_ = v.shape
                    _dict[k] = v.reshape(-1,*shape_[2:])

        for metric_fn in self.metrics_fns:
            metrics_dict.update(
                metric_fn(
                gt_dict=gt_dict,
                pred_dict=pred_dict,
                ))

        return metrics_dict


    def restore_model(
        self,
        restore_file,
        off_scheduler=False,
        ):
        restore_path = os.path.join(self.exp_dir, restore_file)
        assert os.path.exists(restore_path)
        checkpoint = utils.load_checkpoint(
            restore_path,
            self.model,
            self.optimizer,
            None if off_scheduler else self.scheduler
            )
        epoch = checkpoint['epoch']
        return epoch


    def save_model(
        self,
        epoch,
        save_file='last.pt',
        ):
        save_path = os.path.join(self.exp_dir, save_file)
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optim_dict' : self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
            },
            save_path)
