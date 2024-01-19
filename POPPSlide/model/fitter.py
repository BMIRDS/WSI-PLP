from pathlib import Path
import os
import re
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import shutil
from tqdm import tqdm

from model.helper import (
    EarlyStopping, AverageMeter, ProgressMeter, ModelEvaluation, calculate_metrics)

# ----------------
# Helper functions
# ----------------
def unpack_sample(sample, device):
    '''
    Unpack sample to and send the images and label to device.
    '''
    imgs, ids, targets = sample
#     imgs, ids, targets = list(
#         map(lambda x: sample[x], [0,1,2]))
    if isinstance(imgs, torch.Tensor):
        imgs = imgs.to(device)
    if isinstance(ids, torch.Tensor):
        ids = ids.to(device)
    if isinstance(targets, torch.Tensor):
        targets = targets.to(device)
    return imgs, ids, targets


def reshape_img_batch(img_batch, crop_size, need_permute: bool):
    '''
    Reshape image tensor to [batch_size*num_patches, num_channels, crop_size, crop_size]
    '''
    if need_permute:  #NOTE: SHUAI. EXTEND HYBRIDFITTER
        return img_batch.\
        view(img_batch.size(0), 3, -1, crop_size, crop_size).\
        permute(0, 2, 1, 3, 4).\
        contiguous().\
        view(-1, 3, crop_size, crop_size)
    else:
        return img_batch.view(-1, 3, crop_size, crop_size)


def format_results(res):
    '''
    Format logging
    '''
    line = ""
    for key in sorted(res):
        val = res[key]
        if isinstance(val, str) or isinstance(val, int):
            fmt = "%s: %s\t"
        else:
            fmt = "%s: %8.6f\t"
        line += (fmt % (key, val))
    return line


class HybridFitter:
    '''
    Helper class for scheduling training and evaluation.
    model: model to train/evaluate
    dataloader: Dataloader class
    writer: logging the result
    args: arguments passed from configuration file
    timestr: name of the experiment
    loss_function: loss function for training
    '''
    def __init__(
            self,
            model,
            dataloader,  #NOTE: ADDITION
            checkpoint_to_resume='',
            writer=None,
            args=None,
            timestr='',
            model_name='model',
            loss_function=None):
        self.writer = writer
        self.args = args
        self.criterion = loss_function
        self.model = model
        self.device = self.model.device
        self.reset_optimizer()
        self.dataloaders = dataloader
        self.meta_df = {}
        self.es = EarlyStopping(patience=self.args.patience, mode='max')
        self.timestr = timestr
        self.meta_ds = {}
        self.model_name = model_name
        self.checkpoint_to_resume = checkpoint_to_resume
        self.best_metric = 0

        self.current_epoch = 1
        if len(checkpoint_to_resume):
            self.resume_checkpoint()

        metrics = {
            'classification': 'auc',
            'survival': 'c-index',
            'regression': 'r2'
        }
        self.metric = metrics[self.args.outcome_type]
        self.need_permute = args.config is None

    def resume_checkpoint(self):
        '''
        Load pretrained model from last time training.
        '''
        ckp = torch.load(self.checkpoint_to_resume)
        self.writer['meta'].info("Loading model checkpoints ... Epoch is %s" % ckp['epoch'])
        self.model.load_state_dict(ckp['state_dict_model'])
        self.optimizers['adam'].load_state_dict(ckp['state_dict_optimizer_adam'])
        self.optimizers['sgd'].load_state_dict(ckp['state_dict_optimizer_sgd'])
        self.schedulers['adam'].load_state_dict(ckp['state_dict_scheduler_adam'])
        self.schedulers['sgd'].load_state_dict(ckp['state_dict_scheduler_sgd'])
        self.current_epoch = ckp['epoch'] + 1


    def reset_optimizer(self):
        '''
        Initialize model optimization and learning rate schedulers
        '''
        conv_params = []
        for name, param in self.model.backbone.named_parameters():
            if re.search('fc', name):
                pass
            else:
                if param.requires_grad:
                    conv_params.append(param)
        self.optimizers = {}

        self.optimizers['adam'] = optim.Adam([
            {
                'params': conv_params,
                'lr': self.args.lr_backbone,
                'weight_decay': self.args.wd_backbone
            },
            {
                'params': self.model.head.parameters(),
                'lr': self.args.lr_head,
                'weight_decay': self.args.wd_head
            }])

        self.optimizers['sgd'] = optim.SGD(
            self.model.head.parameters(),
            lr=self.args.lr_head,
            weight_decay=self.args.wd_head)

        self.schedulers = {}
        self.schedulers['adam'] = lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizers['adam'], 
            T_0=self.args.cosine_anneal_freq, 
            T_mult=self.args.cosine_t_mult)
        self.schedulers['sgd'] = lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizers['sgd'], 
            T_0=self.args.cosine_anneal_freq, 
            T_mult=self.args.cosine_t_mult)

    def train(self, epoch=0):
        '''Model Training of the current epoch'''
        
        self.writer['meta'].info('Training from step %s' % epoch)
        # training phase
        self.model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        perfs = AverageMeter(self.metric, ':5.4f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':5.4f')
        progress = ProgressMeter(
            len(self.dataloaders['train']),
            [batch_time, data_time, losses, perfs],
            prefix="Epoch: [{}]".format(epoch),
            writer=self.writer['meta'],
            verbose=True)
        end = time.time()

        eval_t = ModelEvaluation(
            outcome_type=self.args.outcome_type,
            loss_function=self.criterion,
            mode='train',
            device=self.device,
            timestr=self.timestr)

        for group in self.optimizers['adam'].param_groups:
            current_lr = group['lr']
            self.writer['meta'].info("Learning rate is %s" % (current_lr))
        self.writer['meta'].info('-' * 30)

        # train over all training data
        for i, sample in enumerate(self.dataloaders['train']):
            train_imgs, train_ids, train_targets = unpack_sample(
                sample, self.device)
            train_imgs = reshape_img_batch(
                train_imgs, self.args.crop_size, self.need_permute)

            nbatches = train_imgs.size(0) // (self.args.num_patches * self.args.num_crops) # batch size

            data_time.update(time.time() - end)

            # forward and backprop
            self.optimizers['sgd'].zero_grad()
            with torch.set_grad_enabled(True):
                ppi = self.args.num_patches * self.args.num_crops # currently only tested num_crops=1
                model_outputs = self.model(train_imgs, ppi)
                train_preds = model_outputs['pred']

                if self.need_permute:
                    train_targets = train_targets.view(nbatches, self.args.num_patches, -1)[:, 0, :]
                train_loss = self.criterion.calculate(train_preds, train_targets)
                train_loss.backward()

                eval_t.update(
                    {
                        "id": train_ids, ### revisit, currently not updating the id of the images
                        "preds": train_preds,
                        "targets": train_targets
                    }
                )

                self.optimizers['adam'].step()
                self.optimizers['adam'].zero_grad()

                all_head_params = torch.cat([x.view(-1)
                                             for x in list(self.model.head.parameters())])
                
                # regularization, not tested, currently l1 and l2 are set to 0
                # TODO: commented out currently to test, revisit later
#                 l1_regularization = self.args.l1 * torch.norm(all_head_params.view(-1), 1)
#                 l2_regularization = self.args.l2 * torch.norm(all_head_params.view(-1), 2).pow(2)
#                 loss_regularization = l1_regularization + l2_regularization

#                 loss_regularization.backward()
#                 self.optimizers['sgd'].step()

            # update metrics
            losses.update(train_loss.item(), nbatches)
         
            train_metrics = calculate_metrics(
                train_preds.data.cpu().numpy(),
                train_targets.data.cpu().numpy(),
                outcome_type=self.args.outcome_type)
            perfs.update(train_metrics[self.metric], nbatches)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % self.args.log_freq == 0:
                progress.display(i + 1)
        
        train_res = eval_t.evaluate()
        train_res['epoch'] = epoch
        train_res['mode'] = 'train'
        return train_res

    def evaluate(self, epoch=0):
        '''Model evaluation of the current epoch'''

        self.writer['meta'].info('Starting evaluation')
        # validation phase
        self.model.eval()
        eval_v = ModelEvaluation(
            outcome_type=self.args.outcome_type,
            loss_function=self.criterion,
            mode='val',
            device=self.device,
            timestr=self.timestr)

        # forward prop over all validation data
        for i, sample in enumerate(tqdm(self.dataloaders['val'])):

            val_imgs, val_ids, val_targets = unpack_sample(sample, self.device)
            val_imgs = reshape_img_batch(val_imgs, self.args.crop_size, self.need_permute)

            nbatches = val_imgs.size(0) // (self.args.num_val * self.args.num_crops)
            
            # forward
            with torch.set_grad_enabled(False):
                val_preds = self.model(val_imgs, self.args.num_val*self.args.num_crops)['pred']
                if self.args.config is None:
                    val_targets = val_targets.view(nbatches, self.args.num_val, -1)[:, 0, :]
                    val_ids = val_ids.view(nbatches, self.args.num_val, -1)[:, 0, :]
                eval_v.update(
                    {
                        "id": val_ids,
                        "preds": val_preds,
                        "targets": val_targets
                    }
                )
        
        # Save the evaluation result of current epoch
        val_res = eval_v.evaluate()
        val_res['epoch'] = epoch
        val_res['mode'] = 'val'
        save_path = os.path.join("predictions",self.model_name,"%04d.csv" % epoch)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  #FIXME: PATHLIB
        eval_v.save(save_path)
        return val_res

    def fit_epoch(self, epoch=0):
        '''
        Helpter function to fit the model for current epoch and save model
        '''
        train_res = self.train(
            epoch=epoch)
        self.writer['data'].info(format_results(train_res))
        self.schedulers['adam'].step()
        self.schedulers['sgd'].step()
        
        
        val_res = self.evaluate(
            epoch=epoch)

        self.writer['data'].info(format_results(val_res))
        
        # Choose the evaluation metrics for outcome type to optimize
        if self.args.outcome_type == 'survival':
            performance_measure = torch.tensor(val_res['c-index'])
        elif self.args.outcome_type == 'classification':
            performance_measure = torch.tensor(val_res['auc'])
        elif self.args.outcome_type == 'regression':
            performance_measure = torch.tensor(val_res['r2'])


        # Save best performed model and at every save_interval
        is_best = False
        if performance_measure > self.best_metric:
            self.best_metric = performance_measure
            print("New best result: %6.4f" % self.best_metric)
            is_best = True

        self.save_checkpoint(
            epoch=epoch,
            is_best=is_best,
            save_freq=self.args.save_interval,
            checkpoints_folder=self.checkpoints_folder
            )
        
        # Early stopping
        # TODO: not tested, revisit
        if epoch >= self.args.epochs:
            if self.es.step(performance_measure):
                return 1  # early stop criterion is met, we can stop now
        return 0

    def save_checkpoint(
            self,
            epoch,
            is_best,
            save_freq,
            checkpoints_folder):

        state_dict = {
            'epoch': epoch,
            'state_dict_model': self.model.state_dict(),
            'state_dict_optimizer_adam': self.optimizers['adam'].state_dict(),
            'state_dict_optimizer_sgd': self.optimizers['sgd'].state_dict(),
            'state_dict_scheduler_adam': self.schedulers['adam'].state_dict(),
            'state_dict_scheduler_sgd': self.schedulers['sgd'].state_dict(),
        }
        # remaining things related to training
        os.makedirs(checkpoints_folder, exist_ok=True)  #FIXME: PATHLIB
        epoch_output_path = os.path.join(checkpoints_folder, "LAST.pt")
        torch.save(state_dict, epoch_output_path)

        if is_best:
            print("Saving new best result!")
            fname_best = Path(checkpoints_folder) / "BEST.pt"
            if os.path.isfile(fname_best):
                os.remove(fname_best)
            shutil.copy(epoch_output_path, fname_best)

        if epoch % save_freq == 0:
            print("Saving new checkpoints!")
            shutil.copy(epoch_output_path, Path(checkpoints_folder) / ("%04d.pt" % epoch)) #FIXME F STRING

    def fit(self, checkpoints_folder='checkpoints'):
        '''Model fitting'''
        
        self.checkpoints_folder = checkpoints_folder

        for epoch in range(self.current_epoch, self.args.epochs + 1):
            return_code = self.fit_epoch(
                epoch=epoch)
            if return_code: # early stopping
                break
