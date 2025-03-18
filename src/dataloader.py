import logging
import numpy as np
import torch
from functools import partial
from multiprocessing import Pool
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import torchvision.transforms as transforms
from utils import load_image, load_test_image


class RegionizedDataset(Dataset):
    def __init__(self, csv_file, transform, images_dir: str, mask_dir: str , which_index: str ,which_class:None, train_or_test="train"):
        self.data_all = pd.read_csv(csv_file).dropna()
        self.data_all["region_index"] = self.data_all["region_index"].astype(int)
        self.which_class_gv = 10 if which_class=="forest" else 80 if which_class=="water" else 40 if which_class=="crop" else 30 if which_class=="grass" else None
        # self.data_all = self.data_all[self.data_all[f"%_{self.which_class_gv}"]>10] # This is to have images with enough amount of interesting classes inside

        self.transforms = transforms.Compose([transforms.CenterCrop(256)])
        self.data = self.data_all.reset_index(drop=True)[:]
            
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.ids = self.data["point_index"]



    def __len__(self):
        return len(self.data)


    def __getitem__(self, ids):

        name = str(self.data.point_index[ids])
        region_index = self.data.region_index[ids]
        # weight = self.data.weight[ids]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0], mask=True,which_class_gv=self.which_class_gv)
        img = load_image(img_file[0], mask=False, which_class_gv=self.which_class_gv)

        return {
            'image':  self.transforms(torch.as_tensor(img.copy())).float().contiguous(),
            'mask':  self.transforms(torch.as_tensor(mask.copy())).long().contiguous(),
            'region_index': torch.as_tensor(region_index).long().contiguous(),
            # 'weight': torch.as_tensor(weight).float().contiguous(),
        }




# if __name__ == "__main__":
    
#     import hydra
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.load('./configs/config.yaml')

#     dl = hydra.utils.instantiate(cfg.dataset)
#     batch = dl.__getitem__(0)
#     print(batch)

#     pass
