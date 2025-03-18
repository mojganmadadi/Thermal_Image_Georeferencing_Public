import logging
import hydra
import os
from tqdm import tqdm
import wandb
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torch.nn.functional as F
import pandas as pd
# from evaluate import evaluate, evaluate_withAddedLAyer
import matplotlib.pyplot as plt
class TrainModel:

    def __init__(self, config):
        
        self.config = config
        self.learning_rate = config.lr
        self.batch_size = config.batch_size
        self.nEpochs = config.nEpochs
        self.num_regions = config.num_regions
        self.global_step = 0
        self.val_percent = 0.1
        self.save_checkpoint = True
        self.img_scale= 1
        self.amp = False
        self.gradient_clipping = 1.0
        self.dir_checkpoint = config.dir_checkpoint
        self.resume_training = config.resume_training
        self.loss = 0
        self.visualize =True
        self.which_index = config.which_index
        # self.predict_at_the_end= config.predict_at_the_end
        self.which_class= config.which_class
        test_batchsize= 16
       
        
        # Device check
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')

        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(config.seed)

        # loading datasets 
        dataset = hydra.utils.instantiate(config.dataset)
        testset = hydra.utils.instantiate(config.test_dataset)

        # self.n_val = 0
        # self.n_train = len(dataset)
        # train_set = dataset

        self.n_val = int(len(dataset) * self.val_percent)
        print(len(dataset))
        self.n_train = len(dataset) - self.n_val
        # self.n_test = len(testset)
        train_set, val_set = random_split(dataset, [self.n_train, self.n_val], generator=torch.Generator().manual_seed(0))
        

        self.train_loader = DataLoader(train_set,
                                 shuffle=True,
                                 batch_size=self.batch_size,
                                 num_workers=8,
                                 pin_memory=True,)
        self.val_loader = DataLoader(val_set,
                                 shuffle=True,
                                 batch_size=self.batch_size,
                                 num_workers=8,
                                 pin_memory=True,)

        self.model = hydra.utils.instantiate(config.model).to(device=self.device)

        if self.resume_training:
            self.resume_from_epoch = np.array(
                [i.split('.pth')[0].split('_epoch')[-1] for i in os.listdir(self.dir_checkpoint)]).astype(np.int32).max()+1
            cpt = torch.load(str('{}/checkpoint_epoch{}.pth'.format(self.dir_checkpoint, self.resume_from_epoch-1)))
            self.model.load_state_dict(cpt)
        else:
            self.resume_from_epoch = 0
        self.criterion = hydra.utils.instantiate(config.loss).to(device=self.device)
        self.optimizer = hydra.utils.instantiate(
           config.optimizer,
           params=self.model.parameters())

        # self.scheduler = hydra.utils.instantiate(config.scheduler, optimizer=self.optimizer)
        self.evaluate = hydra.utils.instantiate(config.evaluate)

        # (Initialize logging)
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.experiment =  wandb.init(project='ThermalImageGeoreferencing', resume='allow',entity="mojganmadadi", name=self.dir_checkpoint)
        self.experiment.config.update(
            dict(epochs=self.nEpochs, batch_size=self.batch_size, learning_rate=self.learning_rate,
                val_percent=self.val_percent, save_checkpoint=self.save_checkpoint, amp=self.amp)
        )

        logging.info(f'''Starting training:
            Epochs:          {self.nEpochs}
            Batch size:      {self.batch_size}
            Learning rate:   {self.learning_rate}
            Training size:   {self.n_train}
            Validation size: {self.n_val}
            Checkpoints:     {self.save_checkpoint}
            Device:          {self.device.type}
        ''')
                # Test size:       {self.n_test}
    def fit(self):
        numberOfBatches = 0
        for epoch in range(self.resume_from_epoch, self.nEpochs + 1):
            self.model.train()
            self.epoch_loss = 0
            with tqdm(total=self.n_train, desc=f'Epoch {epoch}/{self.nEpochs}', unit='img') as pbar:
                for count, batch in enumerate(self.train_loader):

                    images, true_masks, region_index = batch['image'], batch['mask'], batch['region_index']
                    images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=self.device, dtype=torch.long)
                    region_index = region_index.to(device=self.device, dtype=torch.long)
                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        self.model = self.model.to(device=self.device)
                        masks_pred = self.model(images, region_index)
                        self.loss = self.criterion(masks_pred, true_masks.float())
                            

                    self.optimizer.zero_grad()
                    # self.grad_scaler.scale(self.loss).backward()
                    self.loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    # self.grad_scaler.step(self.optimizer)

                    self.optimizer.step()

                    # self.grad_scaler.update()
                    # self.scheduler.step()
                    # self.grad_scaler.update()
                    pbar.update(images.shape[0])
                    self.global_step += 1
                    self.epoch_loss += self.loss.item()
                    self.experiment.log({
                        'train loss': self.loss.item(),
                        'step': self.global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': self.loss.item()})

                # if epoch%10==0:
                #     validator_on_test = self.evaluate(
                #         model=self.model,
                #         dataloader=self.val_loader,
                #         device=self.device)
                #     val_iou, val_loss, val_f1 = validator_on_test.evaluate()
                

                # self.model.train()
                # self.optimizer.zero_grad()
                pred_to_show =  torch.sigmoid(masks_pred)[0].float().cpu() 
                self.experiment.log({
                    'learning rate': self.optimizer.param_groups[0]["lr"],
                    # 'Mean iou on complete test set': val_iou,
                    # 'Loss of complete test set': val_loss,
                    # 'f1 score of complete test set': val_f1,

                    'images': wandb.Image(images[0,0,:,:].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(pred_to_show),
                    },
                    'step': self.global_step,
                    'epoch': epoch,
                })
            # if (self.predict_at_the_end) & (epoch==self.nEpochs):


            if (self.save_checkpoint) & (epoch % 4 == 0):
                Path(self.dir_checkpoint).mkdir(parents=True, exist_ok=True)
                # state_dict = self.model.state_dict()
                to_save = {'epoch':epoch,
                           'model_state_dict':self.model.state_dict(),
                           'optimizer_state_dict':self.optimizer.state_dict(),
                           'loss':self.epoch_loss,
                           }
                torch.save(to_save, str('{}/checkpoint_with_Val_epoch_{}.pth'.format(self.dir_checkpoint, epoch)))
                logging.info(f'Checkpoint {epoch} saved!')



class trainNoModification(TrainModel):

    def __init__(self, config):
        super().__init__(config)
        
    def fit(self):
        numberOfBatches = 0
        for epoch in range(self.resume_from_epoch, self.nEpochs + 1):
            self.model.train()
            self.epoch_loss = 0
            with tqdm(total=self.n_train, desc=f'Epoch {epoch}/{self.nEpochs}', unit='img') as pbar:
                for count, batch in enumerate(self.train_loader):

                    images, true_masks, region_index = batch['image'], batch['mask'], batch['region_index']
                    images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=self.device, dtype=torch.long)
                    region_index = region_index.to(device=self.device, dtype=torch.long)
                    with torch.autocast(self.device.type if self.device.type != 'mps' else 'cpu', enabled=self.amp):
                        self.model = self.model.to(device=self.device)
                        masks_pred = self.model(images)
                        self.loss = self.criterion(masks_pred, true_masks.float())
                            

                    self.optimizer.zero_grad()
                    # self.grad_scaler.scale(self.loss).backward()
                    self.loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
                    # self.grad_scaler.step(self.optimizer)
                    self.optimizer.step()
                    # self.grad_scaler.update()
                    # self.scheduler.step()
                    pbar.update(images.shape[0])
                    self.global_step += 1
                    self.epoch_loss += self.loss.item()
                    self.experiment.log({
                        'train loss': self.loss.item(),
                        'step': self.global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': self.loss.item()})
                if epoch%10==0:
                    validator_on_test = self.evaluate(
                        model=self.model,
                        dataloader=self.val_loader,
                        device=self.device)
                    val_iou, val_loss, val_f1 = validator_on_test.evaluate()
                self.model.train()
                pred_to_show =  torch.sigmoid(masks_pred)[0].float().cpu() 
                self.experiment.log({
                    'learning rate': self.optimizer.param_groups[0]["lr"],
                    'Val iou': val_iou,
                    'Val loss': val_loss,
                    'Val f1': val_f1,
                    'images': wandb.Image(images[0,0,:,:].cpu()),
                    'masks': {
                        'true': wandb.Image(true_masks[0].float().cpu()),
                        'pred': wandb.Image(pred_to_show),
                    },
                    'step': self.global_step,
                    'epoch': epoch,
                })
            

            if (self.save_checkpoint) & (epoch % 5 == 0):
                Path(self.dir_checkpoint).mkdir(parents=True, exist_ok=True)
                # state_dict = self.model.state_dict()
                to_save = {'epoch':epoch,
                           'model_state_dict':self.model.state_dict(),
                           'optimizer_state_dict':self.optimizer.state_dict(),
                           'loss':self.epoch_loss,
                           }
                torch.save(to_save, str('{}/checkpoint__with_Val_epoch_{}.pth'.format(self.dir_checkpoint, epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
