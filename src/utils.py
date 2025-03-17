import numpy as np
import rasterio

def load_image(filename, mask=None, which_class_gv=None):
    with rasterio.open(filename) as data:
        image = data.read([1])
    if not mask:
        image = np.clip(image/40000,0,1) # Adjust this 40000 to the maximum value in your data
    elif which_class_gv!=200:
        image[image==which_class_gv]=1
        image[image!=1]=0
    else:
        image[image==10]=1 # those are based on ESA world cover
        image[image==30]=2
        image[image==40]=3
        image[image==80]=4
        image[(image!=1) & (image!=2) & (image!=3) & (image!=4)]=0
    return image


def load_test_image(filename, mask=None):
    with rasterio.open(filename) as data:
        image = data.read([1])

    if not mask:
        image = np.clip(image/40000,0,1)
    else:
        image[image==10]=1
        image[image==30]=2
        image[image==40]=3
        image[image==80]=4
        image[(image!=1) & (image!=2) & (image!=3) & (image!=4)]=0
    return image
