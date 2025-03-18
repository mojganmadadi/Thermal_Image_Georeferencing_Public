# Thermal_Image_Georeferencing

The repository contains the code for training and testing land cover classifiers namely cropland, water, tree cover, and grasslands and using them to geolocate test data.
The data folder containing train and test images as well their corresponding masks and metadata should be as follows:

```
data/  
├── test/  
│   ├── imgs/  
│   ├── masks/  
├── train/  
│   ├── imgs/  
│   ├── masks/  
Thermal_Image_Georeferencing_Public/  
├── Dockerfile  
├── README.md  
├── requirements.txt  
├── src/  
│   ├── dataloader.py  
│   ├── evaluate.py  
│   ├── Georeference.ipynb  
│   ├── main.py  
│   ├── test.py  
│   ├── train_val.py  
│   ├── unet_model.py  
│   ├── utils.py  
│   ├── Checkpoints/  
│   ├── configs/  
│   │   └── config.yaml  
│   └── wandb/
```

                
To reproduce the result without running into package problems, you can use the docker file to create the same image and use the container to run the code. 

Also, the repo uses [Weights and Biases](https://wandb.ai/site/) to monitor the model performance and evaluation metrics.

The repo follows [Hydra](https://hydra.cc/docs/intro/) template for improved scalability of the project.

For training, the config file can be modified to `train_or_test: "train"` and the training can be run by `python main.py`.
