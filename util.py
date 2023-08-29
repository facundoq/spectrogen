import numpy as np

def pdf_to_samples(x,y,n_samples,smoothing=0.1,jitter=0.2):
    y=y+smoothing
    y/=y.sum()
    samples = []
    for xi,yi in zip(x,y):
        nx = int(np.round(yi*n_samples)[0])
        samples+= [xi+np.random.normal(scale=jitter) for i in range(nx)]
    print(len(samples))
    return np.array(samples)

def sample_pdf(x,y,n):
    y = np.cumsum(y)
    m = len(y)
    s = np.zeros(n)
    probs = np.random.uniform(size=n)
    for j,p in enumerate(probs):
        i=0
        while i<m and y[i]<p:
            i+=1
        s[j]=x[i-1]
    return s
    