from scipy.signal import find_peaks,savgol_filter
from scipy.ndimage import gaussian_filter
import numpy as np

from abc import abstractmethod, ABC
class Step(ABC):

    @abstractmethod
    def __call__(self,) :
        pass

class Pipeline:

    def __init__(self,steps:list[Step]) -> None:
        self.steps=steps

    def __call__(self, inputs):

        for step in self.steps:
            if type(inputs) is tuple:
                inputs = step(*inputs)
            else:
                inputs = step(inputs)
        return inputs

from scipy.signal import savgol_filter
class Image2SignalDistribution(Step):
    def __call__(self,image):
        n,m=image.shape
        x = np.arange(n)
        y = (image.max(axis=1)*0.2+image.mean(axis=1)*0.8)

        # topn = int(n*0.2)
        # y = np.argpartition(image,-topn,axis=1)
        # print(y.shape)
        # y=y[:,:-topn]
        # print(y.shape)
        # y=y.mean(axis=1)
        # print(y.shape)
        y = y -y.min()
        y = y/y.sum()
        
        #smoothing
        
        return x,y
    
class SavgolFilter(Step):
    def __init__(self,window_size_percentage,polyorder) -> None:
        self.window_size_percentage = window_size_percentage
        self.polyorder = polyorder
    def __call__(self,x,y):
        window_size_percentage=0.01
        window_size = int(len(y)*window_size_percentage)
        y = savgol_filter(y,window_size,self.polyorder)
        y = y/y.sum()
        return x,y
    

class BackgroundRemoval(Step):
    def __init__(self,spread_factor_multiplier=40) -> None:
        self.spread_factor_multiplier=spread_factor_multiplier

    def __call__(self,image):
        spread_factor = image.std()*self.spread_factor_multiplier
        image = image-image.mean()/spread_factor
        image[image<0]=0
        image/=image.max()
        image[image<0.3]=0
        # image[image>0.9]=1
        image/=image.max()
        image = gaussian_filter(image, sigma=(4,20))
        image/=image.max()
        return image    
    

class FindPeaks(Step):
    def __init__(self,distance_percentage,height_factor,prominence_factor):
        self.distance_percentage=distance_percentage
        self.height_factor=height_factor
        self.prominence_factor=prominence_factor
        
    def __call__(self,x,y):
        y = y
        n, = x.shape
        prominence = y.mean()/5
        
        peaks, _ = find_peaks(y, distance=max(10,n*self.distance_percentage),height=y.mean()*self.height_factor,prominence=y.std()* self.prominence_factor)
        return peaks,x,y
    