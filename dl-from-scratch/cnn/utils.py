import os 
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from typing import Union, Tuple, List


main_data_path = r"C:\Users\ayhan\Desktop\ml-collection\data"
ten_animals_dataset_path = os.path.join(main_data_path, "ten_animals","raw-img")

def get_random_image(animal:str, img_format:str="jpg"):
    animal_path = os.path.join(ten_animals_dataset_path, animal)
    animal_images = glob.glob(os.path.join(animal_path, "*"))
    selected_animal_img = animal_images[np.random.randint(0,len(animal_images))]
    img = plt.imread(selected_animal_img)
    return img


class ManualKernelConvolution:
    def __init__(self, image:Union[np.ndarray,str],):
        if isinstance(image, str):
            try:
                self.image = plt.imread(image)
            except:
                raise Exception("Invalid image path")
        else:
            self.image = image
        print("loaded image:")
        plt.imshow(self.image, cmap="gray")
        self.n_channels = self.image.shape[2]
        self.kernels = []
        self.n_kernels = 0
        self.channel_map = {
            0:"Red",
            1:"Green",
            2:"Blue"
        }

    def convolve(self,kernels:List[Tuple[str,np.ndarray]]):
        self.kernels = kernels # a kernel has a name and a numpy array
        self.n_kernels = len(kernels)
        grid_size = np.math.ceil(np.math.sqrt(self.n_kernels)) # square grid
        for i in range(self.n_channels):
            plt.figure(figsize=(12,8))
            plt.suptitle(f"applied convolution for channel {self.channel_map[i]}")
            plt.subplots_adjust(hspace=0.5)
            for idx,(name, kernel) in enumerate(self.kernels):
                plt.subplot(grid_size, grid_size, idx+1)
                conv_img = convolve2d(self.image[:,:,i], kernel, mode="same")
                plt.imshow(conv_img, cmap="gray")
                plt.title(name)
            plt.show()
