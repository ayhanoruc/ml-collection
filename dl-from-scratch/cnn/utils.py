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
        # now lets holdout 10% of the self.all_img_paths for validating & 10% for testing (make sure they are balanced as well)
        n_samples = int(0.1*len(self.all_img_paths))
        print("n_samples",n_samples)
        self.val_img_paths,self.val_labels = self.select_balanced_batch_indices(n_samples)
        remove_mask = ~np.isin(self.all_img_paths,self.val_img_paths)
        self.all_img_paths = self.all_img_paths[remove_mask]
        self.test_img_paths,self.test_labels = self.select_balanced_batch_indices(n_samples)
        remove_mask = ~np.isin(self.all_img_paths,self.test_img_paths)
        self.all_img_paths = self.all_img_paths[remove_mask]
        self.network_input_x, self.network_input_y = self.network_input_size
        
    def return_val_test_batches(self,):
        val_tuple = self.prepare_batch(self.val_img_paths),self.val_labels
        test_tuple = self.prepare_batch(self.test_img_paths),self.test_labels
        return val_tuple,test_tuple

    def __iter__(self):
        return self

    def __next__(self):
        self.batch_img_paths, self.batch_labels = self.select_balanced_batch_indices(self.batch_size)
        return self.prepare_batch(self.batch_img_paths)


    def prepare_batch(self,batch_img_paths):
        self.batch_matrix = np.zeros((self.network_input_x, self.network_input_y, 3,len(batch_img_paths)))
        for i, img_path in enumerate(batch_img_paths):
            img = Image.open(img_path)
            img = img.resize((self.network_input_x, self.network_input_y))
            img = np.array(img)
            if len(img.shape) == 2: # if the image is grayscale
                img = np.stack((img,)*3, axis=-1)
            self.batch_matrix[:, :, :, i] = img # put the image in the batch matrix
        # then at the end make sure its in 8bits representation
        self.batch_matrix = self.batch_matrix.astype("uint8")
        return self.batch_matrix
        
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
    

    def select_balanced_batch_indices(self,batch_size) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects a balanced batch of indices, ensuring equal representation of each class.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the batch image paths and labels.
        """
        num_classes = len(self.data_map)
        samples_per_class = batch_size // num_classes

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



class ManualConvLayer:
    def __init__(self,kernel_shape=(3,3),kernel_count:int=5,stride:int=1,padding:int=0) -> None:
        self.kernel_x , self.kernel_y = kernel_shape
        self.kernel_count = kernel_count # represents the number of output feature maps (or channels)
        self.stride = stride
        self.padding = padding
        # here kernels and their corresponding biases are learned parameters
        # He initialization
        fan_in = self.kernel_x * self.kernel_y * kernel_count
        stddev = np.sqrt(2 / fan_in)
        self.weights = np.random.randn(self.kernel_x, self.kernel_y,self.kernel_count) * stddev
        self.biases = np.zeros((1,self.kernel_count))

    def forward(self,img_batch:np.ndarray) -> np.ndarray:
        img_x,img_y,img_channels,batch_size = img_batch.shape

        self.output_x = (img_x - self.kernel_x + 2 * self.padding) // self.stride + 1
        self.output_y = (img_y - self.kernel_y + 2 * self.padding) // self.stride + 1
        self.output =  np.zeros((self.output_x, self.output_y, img_channels, self.kernel_count, batch_size))
        self.weights = np.nan_to_num(self.weights)
        self.biases = np.nan_to_num(self.biases)
        self.padded_image = np.zeros((img_x + 2 * self.padding, img_y + 2 * self.padding, img_channels, self.kernel_count, batch_size))
        
        # padd all batch images for each kernel, think it of as parallel strings.
        # FIXME: The padding is applied for each kernel separately, which is unnecessary and inefficient. Padding should be applied once to the input.
        for k in range(self.kernel_count):
            self.padded_image[self.padding:img_x + self.padding, self.padding:img_y + self.padding, :, k, :] = img_batch

        # TODO: vectorize - parallelize this loop
        for i in range(batch_size):
            padded_img = self.padded_image[:, :, :, :, i]
            for c in range(img_channels):
                for k in range(self.kernel_count):
                    for y in range(self.output_y):
                        for x in range(self.output_x):
                            # adjusting the current sliding window
                            x_start = x * self.stride
                            x_end = x_start + self.kernel_x
                            y_start = y*self.stride
                            y_end = y_start + self.kernel_y
                            current_window = padded_img[x_start:x_end, y_start:y_end, c, k]
                            self.output[x, y, c, k, i] = np.sum(np.multiply(current_window, self.weights[:,:,k])) + self.biases[0,k].astype("float64")

        # now we need to sum the convolutions over all channels: axis=2
        # Summing across the input channels (axis=2) effectively combines the contributions from each input channel for a given kernel and spatial location.
        self.output = self.output.sum(axis=2)
        # now lets save the input as well (for backpropagation)
        self.input = img_batch
        return self.output
    

    def backward(self,outer_deriv:np.ndarray)->np.ndarray:
        
        self.dbiases = np.zeros_like(self.biases)
        self.dweights = np.zeros_like(self.weights)
        x_pad,y_pad,c_pad,k_pad,i_pad = self.padded_image.shape
        dinputs = np.zeros((x_pad,y_pad,c_pad,i_pad)) # we need to go from padded image -> (x,y,c,i) so we need to remove kernel dim.
        x_outer,y_outer= outer_deriv.shape[0],outer_deriv.shape[1]
        padded_img = self.padded_image[:,:,:,0,:] # to remove kernel dim.
        
        for i in range(i_pad):
            current_padded_img = padded_img[:, :, :, i]
            for c in range(c_pad):
                for k in range(self.kernel_count):
                    for y in range(y_outer):
                        for x in range(x_outer):
                            x_start = x * self.stride
                            x_end = x_start + self.kernel_x
                            y_start = y*self.stride
                            y_end = y_start + self.kernel_y
                            current_window = current_padded_img[x_start:x_end, y_start:y_end, c]
                            # # accumulate the differential terms for input and weights
                            dinputs[x_start:x_end, y_start:y_end, c, i] += self.weights[:,:,k]*outer_deriv[x,y,k,i] 
                            self.dweights[:,:,k] += current_window * outer_deriv[x,y,k,i]  
                    # after convolving this kernel, calculate/accumulate the differential term for bias
                    self.dbiases[0,k] += np.sum(np.sum(outer_deriv[:,:,k,i],axis=0),axis=0)
        dinputs = dinputs[self.padding:x_pad-self.padding,self.padding:y_pad-self.padding,:,:]
        return dinputs
        


class PoolingLayer:
    # pooling layer doesnt have a learnable parameter.
    def __init__(self,method:str,kernel_shape:Tuple[int,int]=(3,3),stride:int=1) -> None:
        methods = {
            "average":np.mean, # TODO: actually this requires different implementation
            "max":np.max,
            "min":np.min
        }
        self.method = methods.get(method,np.max)
        self.kernel_shape = kernel_shape
        self.stride = stride

    def forward(self, img_batch:np.ndarray) -> np.ndarray:
        self.img_x, self.img_y, self.img_channels, self.batch_size = img_batch.shape
        print("batch_size",self.batch_size)
        print("img_channels",self.img_channels)
        self.inputs = img_batch
        self.output_x = (self.img_x - self.kernel_shape[0]) // self.stride + 1
        self.output_y = (self.img_y - self.kernel_shape[1]) // self.stride + 1
        self.output = np.zeros((self.output_x, self.output_y, self.img_channels, self.batch_size)) # no need for kernel count since pooling is done per channel
        
        self.values_of_interest = np.zeros((self.kernel_shape[0]*self.kernel_shape[1],2,self.output_x*self.output_y,self.img_channels,self.batch_size))
        self.n_values_of_interest = np.zeros((self.output_x*self.output_y,self.img_channels,self.batch_size))


        # and not added over channels
        padded_image = img_batch
        self.mask = padded_image.copy()
        for i in range(self.batch_size):

            current_padded_img = padded_image[:, :, :, i]
            for c in range(self.img_channels):
                slice_i = 0
                for y in range(self.output_y):
                    for x in range(self.output_x): # this inner loop actually defines a complete slice, and
                        # for each channel-feature map, we expect self.output_x*self.output_y slices.
                        slice_i += 1
                        x_start = x * self.stride
                        x_end = x_start + self.kernel_shape[0]
                        y_start = y*self.stride
                        y_end = y_start + self.kernel_shape[1]
                        current_window = current_padded_img[x_start:x_end, y_start:y_end, c]
                        val = float(self.method(current_window))
                        self.output[x, y, c, i] = val

                        val_x,val_y = np.where(current_window == val)
                        # we can have more than one value of interest
                        for ii, (vx,vy) in enumerate(zip(val_x,val_y)):
                            self.values_of_interest[ii,0,slice_i-1,c,i] = vx
                            self.values_of_interest[ii,1,slice_i-1,c,i] = vy
                        self.n_values_of_interest[slice_i-1,c,i] = ii+1 # for backpropagation

                        # then we have to keep track of the indices that contributed to the max value
                        # for backpropagation
                        self.mask[x_start + val_x, y_start + val_y, c, i] = 1
                # if c == 0:
                #     print(f"for channel {c}, the max slice_i is {slice_i}")
            print(f"FINISHED element {i}/{self.batch_size}")

        return self.output
    
    def backward(self, outer_deriv:np.ndarray) -> np.ndarray:
        x_outer,y_outer,n_channels,batch_size = outer_deriv.shape
        self.dinputs = np.zeros_like(self.inputs)

        for i in range(batch_size):
            for c in range(n_channels):
                slice_i = 0
                for y in range(y_outer):
                    for x in range(x_outer):
                        slice_i += 1
                        x_start = x * self.stride
                        y_start = y*self.stride

                        # how many maxima(actually value of interest determined by the pooling method)
                        # exists in this slice  = n_values
                        n_values = int(self.n_values_of_interest[slice_i-1,c,i])
                        for idx in range(n_values):
                            # then iterate thru it and for modify outer_deriv accordingly
                            val_x = int(self.values_of_interest[idx,0,slice_i-1,c,i])
                            val_y = int(self.values_of_interest[idx,1,slice_i-1,c,i])
                            self.dinputs[x_start+val_x,y_start+val_y,c,i] = outer_deriv[x,y,c,i]
        return self.dinputs
                        

# flattening layer will , for each image in the batch, flatten the image into a 1D array
# of L = W * H * C length, where C is the number of channels : features to our fully connected layer
class Flatten:

    def forward(self, img_batch:np.ndarray) -> np.ndarray:
        # TODO: a more robust way would be getting batch_matrix, batch_size, img_x, img_y, img_channels explicitly
        # then reshaping the batch_matrix to (batch_size, img_x * img_y * img_channels)
        self.batch_size = img_batch.shape[0]
        self.inputs = img_batch
        self.output = img_batch.reshape(self.batch_size,-1 )
        # this requires the batch_size dim to be the first dimension
        # so there is a difference between this and the following implementation!
        # tells the numpy to infer required dimension
        # for the first dimension so that there are to be flattened batch_sized rows. 
        return self.output

    def backward(self, outer_deriv:np.ndarray) -> np.ndarray:
        # no derivatives needes, just re-arrange(reshape) the lower level derivatives
        # into the input shape of the flatten layer
        return outer_deriv.reshape(self.inputs.shape)


class Flatten2: # this is way explicit implementation but doesnt utilize vectorization

    def forward(self, img_batch:np.ndarray) -> np.ndarray:
        self.img_x, self.img_y, self.img_channels,self.batch_size = img_batch.shape
        self.inputs = img_batch
        L = self.img_x * self.img_y * self.img_channels
        self.output = np.zeros((self.batch_size, L))
        for i in range(self.batch_size):
            self.output[i,:] = img_batch[:, :, :,i].reshape((1,L)) # or flatten()
        return self.output
    
    def backward(self, outer_deriv:np.ndarray)->np.ndarray:
        self.dinputs = np.zeros(self.inputs.shape)
        for i in range(self.batch_size):
            self.dinputs[:,:,:,i] = outer_deriv[i,:].reshape(self.img_x,self.img_y,self.img_channels)
        return self.dinputs



class SigmoidAct:
    def forward(self,batch_matrix:np.array)-> np.array:
        # sigmoid maps -inf,inf -> 0,1
        self.output = np.clip(1/(1+np.exp(-batch_matrix)),1e-7,1- 1e-7)
        self.input = batch_matrix
        return self.output
    
    def backward(self,outer_deriv)->np.array:
        inner_deriv = self.output * (1-self.output) # sigmoid x (1-sigmoid)
        self.dinputs = np.multiply(outer_deriv, inner_deriv ) 
        return self.dinputs

class TanhAct:
    # tanh maps -inf,inf -> -1,1

    def forward(self,batch_matrix:np.array)-> np.array:
        self.output = np.tanh(batch_matrix)
        self.input = batch_matrix
        return self.output

    def backward(self,outer_deriv)->np.array:
        inner_deriv = 1 - self.output**2
        self.dinputs = np.multiply(outer_deriv, inner_deriv ) 
        return self.dinputs
    


class SGDOptimizer:
    def __init__(self,lr:0.01,lr_decay_rate:float= None,momentum:float = None):
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.current_learning_rate = lr
        self.iterations = 0
        self.momentum = momentum

    def pre_params_update(self,):
        # monothonic lr decay
        # Rapid Initial Decay, Slow Long-Term Decay
        # Asymptotic Behavior: altough the lr gets closer to zero, it never reaches to zero,
        # ensuring that the optimizer keeps making progress, albeit at a very slow pace, even after many iterations 
        if self.lr_decay_rate:
            self.current_learning_rate = self.learning_rate/(1+self.iterations*self.lr_decay_rate)
    
    def update_params(self,layer):
        # TODO: i may actually define a base abstract class for layer
        """
        - self.momentum: controls how much influence the previous momentum has on the current update.
        - momentum helps to smooth out the weight updates by incorporating information from previous iterations
        - momentum can help the optimizer to escape shallow local minima by allowing it to "roll over"
          small bumps in the loss landscape.
        - by accumulating momentum, the optimizer can move more quickly in consistent directions, 
        leading to faster convergence.
        """
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
            
            weight_updates = self.momentum * layer.weight_momentums -\
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * layer.bias_momentums -\
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        
        else: 
            weight_updates = -self.current_learning_rate* layer.dweights
            bias_updates   = -self.current_learning_rate* layer.dbiases
        
        layer.weights += weight_updates
        layer.biases  += bias_updates

    def post_params_update(self,):
        self.iterations += 1


class DenseLayer:
    # classic fully connected layer
    def __init__(self, n_inputs,n_neurons):
        
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases  = np.zeros((1,n_neurons))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        self.inputs = inputs
        return self.output
    
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dinputs  = np.dot(dvalues,self.weights.T)
        self.dbiases  = np.sum(dvalues, axis = 0, keepdims = True)
        return self.dinputs



class ReluAct:
    def forward(self,batch_matrix:np.array)-> np.array:
        self.output = np.maximum(0,batch_matrix)
        self.input = batch_matrix
        return self.output
    
    def backward(self,outer_deriv)->np.array:
        # filter out the negative values
        self.dinputs = np.multiply(outer_deriv, np.int64(self.output > 0))
        return self.dinputs
    

class SoftmaxAct:
    def forward(self,batch_matrix:np.array)-> np.array:
        # softmax maps -inf,inf -> 0,1 and the sum of the output is 1
        # this is the output of the network
        # for numerical stability, we subtract the maximum value from the input matrix
        self.output = np.exp(batch_matrix - np.max(batch_matrix, axis=1, keepdims=True))
        # the normalization will cancel out the shift effect and the output will not change
        self.output /= np.sum(self.output, axis=1, keepdims=True) # probs.
        self.input = batch_matrix
        return self.output
    
    def backward(self,outer_deriv)->np.array:
        # the derivative of softmax is softmax * (1 - softmax)
        # we can use the derivative of the softmax function to calculate the derivative of the cross-entropy loss
        self.dinputs = np.empty_like(outer_deriv)
        for i, (single_output, single_dvalues) in enumerate(zip(self.output, outer_deriv)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian_matrix, single_dvalues)
        return self.dinputs
    

class CategoricalCrossEntropyLoss:
    """
    go thru basics of information theory:
    - kl-divergence:quantifies how one probability distribution diverges from a second,
      expected probability distribution : https://youtu.be/SxGYPqCgJWM?si=b2Nyse2VcAfMBKaa
      - but this is not a true distance metric, it is not symmetric and does not satisfy the triangle inequality
       neverthless its the backbone of cross-entropy
    - cross-entropy: measures dissimilarity as the average number of bits needed to encode 
        data from one distribution using the code optimized for another distribution.
        Relationship to KL-Divergence:
        D_KL(P || Q) = H(P, Q) - H(P)
        Where H(P, Q) is cross-entropy and H(P) is the entropy of distribution P.
    - categorical cross-entropy: is a specific form of cross-entropy when the true labels are 
            one-hot encoded (e.g., in multi-class classification)
    - Loss = - Σ (y_i * log(p_i)) for i in range(K)
    """
    def forward(self,y_pred:np.array,y_true:np.array)-> np.array:
        self.y_pred = y_pred
        self.y_true = y_true
        self.n_samples = y_pred.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7) # prevent division by 0
        # if even a single zero-confidence prediction is made, the batch loss will be infinite
        # which is disaster :D
        # calculate sample-wise negative log-likelihood 
        #sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        # NOTE: keep in mind that this negative log-likelihood is derived from
        # binomial distribution and maximum likelihood estimation.(log is for computational convenience)
        # and simplifies to negative log of the predicted probability for the correct class 
        # due to one-hot encoding (for both binary and multi-class classification).
        # batch loss
        #self.loss = np.mean(sample_losses)
        if len(y_true.shape) == 1: # sparse
            correct_confidences = y_pred_clipped[range(self.n_samples), y_true]
            #y_true = np.eye(len(y_pred[0]))[y_true]
        elif len(y_true.shape) == 2: # one-hot encoded
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)

        neg_log_likelihoods = -np.log(correct_confidences)

        return neg_log_likelihoods
    
    def backward(self,outer_deriv:np.array)->np.array:
        n_samples = len(outer_deriv)
        if len(self.y_true.shape) == 1:
            n_labels = len(outer_deriv)
            y_true = np.eye(n_labels)[self.y_true]
        self.dinputs = -y_true / outer_deriv / n_samples
        return self.dinputs