import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn

from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score

import pandas as pd
import matplotlib.pyplot as plt

    
class evaluate_iou():
    def __init__(self, model, dataloader,device) -> None:
        self.model = model
        
        self.dataloader = dataloader
        self.num_val_batches = len(dataloader)
        self.device = device
        self.output = 0
        self.df = pd.DataFrame()

    def evaluate(self):
        jaccard = BinaryJaccardIndex().to(device=self.device)
        loss_criterian = torch.nn.BCEWithLogitsLoss().to(device=self.device)
        f1metric = BinaryF1Score().to(device=self.device)
        self.model.eval()
        running_loss = 0.0
        for count, batch in enumerate(tqdm(self.dataloader, total=self.num_val_batches, desc='Validation round', unit='batch', leave=False)):

            image, mask_true, region_index = batch['image'], batch['mask'], batch['region_index']

            image = image.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=self.device, dtype=torch.long)
            region_index = region_index.to(device=self.device, dtype=torch.long)

            # predict the mask
            mask_pred = self.model(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            loss = loss_criterian(mask_pred, mask_true.float())
            jaccard.update((torch.sigmoid(mask_pred) > 0.5).float(), mask_true)
            f1metric.update(mask_pred, mask_true)
            running_loss += loss.item() * image.size(0)

        
        self.model.train()
        return jaccard.compute(), running_loss / len(self.dataloader), f1metric.compute()


class predict_test_set():
    def __init__(self, model, dataloader,device) -> None:
        self.model = model
        self.dataloader = dataloader
        self.num_val_batches = len(dataloader)
        self.device = device
        self.output = 0

    def evaluate(self):
        # jaccard = BinaryJaccardIndex().to(device=self.device)
        # loss_criterian = torch.nn.BCEWithLogitsLoss().to(device=self.device)
        # f1metric = BinaryF1Score().to(device=self.device)
        self.model.eval().to(device=self.device)
        # running_loss = 0.0
        for batch in tqdm(self.dataloader, total=self.num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, region_index = batch['image'], batch['mask'], batch['region_index']
            image = image.to(device=self.device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=self.device, dtype=torch.long)
            region_index = region_index.to(device=self.device, dtype=torch.long)
            mask_pred = self.model(image, region_index)

            ax, fig = plt.subplots(1,3)
            ax[0].imshow(image)
            ax[1].imshow(mask_true)
            ax[2].imshow(mask_pred)

            plt.imshow()
        
        self.model.train()
        return 