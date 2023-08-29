

from scipy.stats import norm
from tqdm.auto import tqdm
import numpy as np


class GMM:

    @classmethod
    def random_initialize(klass,k,x,y,mu=None,sigma=None,pi=None):
        if mu is None:
            indices = np.random.randint(0,high=x.shape[0],size=k)
            mu = x[indices,:]
            # print("Initial mu = ",self.mu)
        if sigma is None:
            sigma = np.ones(k)
            sigma *= np.sqrt(np.mean((x**2)*y)-np.mean(x*y)**2)
                
        if pi is None:
            pi=np.zeros(k)+(1/k)
        # # check dist
        # total = np.zeros(k)
        # for i in range(k):
        #     dist = norm(loc=mu[i],scale=sigma[i])
        #     total[i] = dist.pdf(x).sum()
        
        
        return GMM(mu,sigma,pi)
    
    def __init__(self,mu,sigma,pi):
        self.mu=mu
        self.sigma= sigma
        self.pi=pi

    @property
    def k(self):
        return self.pi.shape[0]
    
    def fit(self,x,y,steps,callbacks=[]):
        
        n, = x.shape
        z = np.zeros( (n,self.k))
        for i in tqdm(range(steps)):
            log_likelihood = self.log_likelihood(x,y)
            # print(self)            
            z = self.e_step(x,y,z)
            for callback in callbacks:
                callback(self,x,y,z,i,steps,log_likelihood)
            #E step
            
            self.m_step(x,y,z)
    def log_likelihood(self,x,y):
        logs = np.zeros( (x.shape[0],self.k))
        for i,d in enumerate(self.get_distributions()):
            logs[:,i]=d.pdf(x)*self.pi[i]*y
        return np.sum(np.log(logs.sum(axis=1)))

    def e_step(self,x,y,z,eps=1e-32):
        

        for i,dist in enumerate(self.get_distributions()):
            val = dist.pdf(x)[:]*y*self.pi[i]
            z[:,i] = val.squeeze()
        zsum = z.sum(axis=1)
        # z[zsum<eps,:]=1/self.k
        # print(np.sum(zsum<eps),x.shape[0])
        #zsum = z.sum(axis=1)
        # print(z.shape,zsum.shape)
        #z[zsum>0]=
        z/=z.sum(axis=1,keepdims=True)
        z[np.isnan(z)]=0
        
        # print(z)
        return z
    
    def get_distributions(self):
        return [norm(loc=mu,scale=sigma) for mu,sigma in zip(self.mu,self.sigma)]
    
    def m_step(self,x,y,z):
        n = len(y)
        self.pi=z.mean(axis=0)
        self.pi/=self.pi.sum()
        
        zsum = z.sum(axis=0)
        self.mu = (z.T @ x) / zsum
        for i in range(self.k):
            #self.mu[i] = (x*z[:,i]).sum()/zsum[i]
            self.sigma[i] = np.sum(((x-self.mu[i])**2)*z[:,i]*y[i])
            self.sigma[i]/= zsum[i]

    def __repr__(self) -> str:
        result = ""
        for i in range(self.k):
            result+= f"Component {i}: π={self.pi[i]:.2f}, μ={self.mu[i]:.2f}, σ={self.sigma[i]:.2f}\n"
        return result
    

    if __name__=="__main__":
        from gmm_density import GMM
        from gmm_density_plot  import GMMPlotCallback
        
        gmm = GMM.random_initialize(3,x,y,mu=peaks.copy())
        # print(f"Initialization:\n {gmm}")
        plot_cb =GMMPlotCallback(plot_z=True)
        gmm.fit(x,y,100,callbacks=[])

