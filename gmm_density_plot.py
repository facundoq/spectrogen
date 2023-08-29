from IPython.display import clear_output
import matplotlib.pyplot as plt
from gmm_density import GMM

class GMMPlotCallback:

    def __init__(self,ax=None,plot_z=False) -> None:
        self.ax=ax
        self.plot_z=plot_z
        self.z_ax=None
    def __call__(self,gmm:GMM,x,y,z,step,steps,log_likelihood):
        # if self.ax is None:
        f,self.ax = plt.subplots(1)
        # self.ax.clear()
        # clear_output(wait=True)    

        self.ax.plot(x,y)
        distributions = gmm.get_distributions()
        for i in range(gmm.k):
            distribution = distributions[i]
            self.ax.plot(x,distribution.pdf(x)*gmm.pi[i],linewidth=4*gmm.pi[i]+0.2,label = f"μ={gmm.mu[i]:.2f},σ={gmm.sigma[i]:.3f},π={gmm.pi[i]:.2f}")
        if gmm.k<6:
            plt.legend()
        plt.title(f"Fit {step+1}/{steps}, -ll={-log_likelihood}")

        if self.plot_z:
            # if self.z_ax is None:
            f2,self.z_ax = plt.subplots(1,gmm.k)
            plt.suptitle("Z")
            for i in range(gmm.k):
                self.z_ax[i].plot(x,z[:,i])
                self.z_ax[i].set_title(f"Component {i}")