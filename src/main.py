import os
import hydra
import glob
import sys
from omegaconf import DictConfig, OmegaConf

import json

@hydra.main(version_base = None, config_path="./configs/", config_name="config.yaml")
def main(config : DictConfig) -> None:
    # print(config)
    #trainer = hydra.utils.instantiate({" _target_" : "train.TrainModel", "_partial_" : True})
    if config.train_or_test =="train":
        trainer = hydra.utils.instantiate(config.trainroutine)
        trainer = trainer(config)
        trainer.fit()


    elif config.train_or_test =="test":
        tester = hydra.utils.instantiate(config.test)
        tester = tester(config)
        tester.predict()
    
if __name__  == "__main__":
    main()