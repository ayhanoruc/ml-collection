import os 
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve2d
from typing import Union, Tuple, List, Dict
from PIL import Image
import glob



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



class ImageBatchGenerator:
    def __init__(self, data_map:Dict[str,Union[str,List[np.ndarray]]], batch_size:int=32, network_input_size:Tuple[int]=(224,224)) -> None:
        self.batch_size = batch_size
        self.network_input_size = network_input_size
        
        self.data_map = data_map
        self.classes = list(data_map.keys())
        self.class_map = {class_name:idx for idx, class_name in enumerate(self.classes)}

        for key in self.classes:
            if isinstance(data_map[key], str):
                if not os.path.exists(data_map[key]) or not os.path.isdir(data_map[key]):
                    raise Exception(f"Invalid folder path for class {key}")
                self.data_map[key] = glob.glob(data_map[key] + "/*")

        self.class_lengths = {class_name:len(data_map[class_name]) for class_name in self.classes}
        self.all_lengths = list(self.class_lengths.values())
        

        self.labels = {class_name:np.zeros(self.class_lengths[class_name]) + self.class_map[class_name] for class_name in self.classes}
        self.all_labels = np.concatenate(list(self.labels.values()))
        self.all_img_paths = np.concatenate(list(self.data_map.values()))

        self.batch_img_paths, self.batch_labels = self.select_balanced_batch_indices()

        self.network_input_x, self.network_input_y = self.network_input_size

        self.batch_matrix = np.zeros((self.network_input_x, self.network_input_y, 3,self.batch_size))
        for i, img_path in enumerate(self.batch_img_paths):
            img = Image.open(img_path)
            img = img.resize((self.network_input_x, self.network_input_y))
            img = np.array(img)
            if len(img.shape) == 2: # if the image is grayscale
                img = np.stack((img,)*3, axis=-1)
            self.batch_matrix[:, :, :, i] = img # put the image in the batch matrix
        # then at the end make sure its in 8bits representation
        self.batch_matrix = self.batch_matrix.astype("uint8")
        
    def visualize_batch(self) -> None:
        # lets create a grid of square
        grid_size = int(np.ceil(np.sqrt(self.batch_size)))
        _, ax = plt.subplots(grid_size, grid_size, figsize=(20, 20))
        for i in range(self.batch_size):
            img = self.batch_matrix[:, :, :, i]
            x = i // grid_size
            y = i % grid_size
            ax[x, y].imshow(img)
            ax[x, y].axis("off")
        plt.show()
    

    def select_balanced_batch_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects a balanced batch of indices, ensuring equal representation of each class.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the batch image paths and labels.
        """
        num_classes = len(self.data_map)
        samples_per_class = self.batch_size // num_classes

        batch_img_paths = []
        batch_labels = []
        # we first ensured the correct number of samples per class, then we randomly select samples from each class
        for class_label, img_paths in self.data_map.items():
            class_indices = np.random.choice(len(img_paths), samples_per_class, replace=False)            
            batch_img_paths.extend([img_paths[i] for i in class_indices])
            batch_labels.extend([class_label] * samples_per_class)

        # shuffle to ensure randomness
        combined = list(zip(batch_img_paths, batch_labels))
        np.random.shuffle(combined)
        batch_img_paths, batch_labels = zip(*combined)

        return np.array(batch_img_paths), np.array(batch_labels)


    def invert_label(self,label:int) -> str:
        try: 
            return self.classes[label]
        except IndexError:
            print("data label out of range, the model knows only the following classes: ", self.classes)
            return None
