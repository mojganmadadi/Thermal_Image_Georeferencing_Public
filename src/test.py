import logging
import hydra
import os
from tqdm import tqdm
import wandb
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex
import pandas as pd
import matplotlib.pyplot as plt
from torchmetrics.functional.classification import binary_f1_score
import unet_model
from matplotlib.colors import ListedColormap


class test:

    def __init__(self, config):
        self.config = config
        self.which_epoch = config.which_epoch_to_evaluate
        self.num_regions = config.num_regions
        self.dir_checkpoint = config.dir_checkpoint
        self.lr = config.lr
        self.where_to_save = config.where_to_save
        os.makedirs(self.where_to_save, exist_ok=True)
        # self.numerator = config.numerator



        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = True

        # loading datasets 
        # dataset = hydra.utils.instantiate(config.dataset)
        testset = hydra.utils.instantiate(config.test_dataset)
        self.test_loader = DataLoader(testset,
                                 shuffle=False,
                                 drop_last=False,
                                 batch_size=1,
                                 num_workers=8,
                                 pin_memory=True,
                                 sampler=None)



        self.model= unet_model.UNet(n_classes=1, n_channels=1)
        self.state_dict = torch.load(os.path.join(self.dir_checkpoint, f"checkpoint__with_Val_epoch_{self.which_epoch}.pth"))
        self.model.load_state_dict(self.state_dict['model_state_dict'])
        self.model.eval()


    def predict(self):
        # df = pd.DataFrame()
        cmp=ListedColormap(['black','green'])
        jaccard = BinaryJaccardIndex().to(device=self.device)
        for count, batch in enumerate(tqdm(self.test_loader, unit="imgs", desc="Predicting images:")):

            images, true_masks, region_index = batch['image'], batch['mask'], batch['region_index']
            images = images.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=self.device, dtype=torch.long)
            region_index = region_index.to(device=self.device, dtype=torch.long)
            self.model = self.model.to(device=self.device)
            masks_pred = self.model(images)
            
            masks_pred = (torch.sigmoid(masks_pred)>.5).float()
            jaccard.update(masks_pred, true_masks.float())
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(images[0].squeeze().cpu().detach().numpy(),cmap="gray")
            ax[0].set_title(f"Thermal Image RI:{region_index[0]}")
            ax[0].set_xticks([]), ax[0].set_yticks([])
            ax[1].imshow(true_masks[0].squeeze().cpu().detach().numpy(),cmap=cmp)
            ax[1].set_title("Ground truth")
            ax[1].set_xticks([]), ax[1].set_yticks([])
            ax[2].imshow(masks_pred[0].squeeze().cpu().detach().numpy(),cmap=cmp)
            ax[2].set_title(f"Regional Prediction")
            ax[2].set_xticks([]), ax[2].set_yticks([])

            plt.savefig(f"tests/{count}.jpg")

