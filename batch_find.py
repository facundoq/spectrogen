from sklearn.linear_model import LinearRegression
from peak_finding import BackgroundRemoval,Image2SignalDistribution,Pipeline,SavgolFilter,FindPeaks
from pathlib import Path
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

signal_pipeline =Pipeline([
    BackgroundRemoval(spread_factor_multiplier=40),
    Image2SignalDistribution(),
    SavgolFilter(window_size_percentage=0.01,polyorder=1),
    FindPeaks(0.2,1.3,2),
])

def find_signal(image,pipeline):
    
    n,m = image.shape
    window_size = 50
    stride = window_size
    def fullrow_windows(image,window_size,stride):
        n,m=image.shape

        for i in range(0,m-window_size+1,stride):
            start = i
            end = start+window_size
            yield image[:,start:end],i

    center = np.zeros(m)
    c = n//2
    for window,i in fullrow_windows(image,window_size,stride):
        peaks,x,y = pipeline(window)
        if len(peaks)>0:
            vals = y[peaks]
            c = peaks[vals.argmax()]
        
        center[i:i+window_size]= c
    return center

folder = Path("/home/facundoq/data/datasets/spectrogram/scans2023/")
results_folder = Path("results")
results_folder.mkdir(exist_ok=True)
file = 10
files = list([x for x in folder.iterdir() if x.is_file()])
files.sort()
for filepath in files:
    print(f"Loading {filepath}")

    image = 1-imread(filepath)
    n,m=image.shape
    if n>m:
        image=image.transpose(1,0)
    

    center = find_signal(image,signal_pipeline)
    x=np.arange(len(center)).reshape(-1,1)
    model=LinearRegression().fit(x, center.reshape(-1,1))
    linear_center = model.predict(x)
    plt.figure(dpi=200)
    plt.imshow(image,cmap="gray")
    plt.plot(center,ls='dotted', linewidth=2, color='red',label="column-wise estimation")
    plt.plot(linear_center, linewidth=1, color='green',label="linear regression")
    plt.legend(bbox_to_anchor=(0.5, 2), loc='upper center')
    output_filepath = results_folder/ (filepath.stem+"_center.png")
    plt.tight_layout()
    plt.savefig(output_filepath)
    plt.close()

