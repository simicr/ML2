""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def task1():
    """ Subtask 1: Approximating Kernels

        Requirements for the plot:
        - the first row corresponds to the task in 1.1 and the second row to the task in 1.2

        for each row:
        - the first subplot should contain the Kernel matrix with each entry K_ij for k(x_i,x_j')
        - the second to fifth subplot should contain the corresponding feature approximation when using 1,10,100,1000 features
    """

    fig, axes = plt.subplots(2, 5)
    fig.set_size_inches(15, 8)
    font = {'fontsize': 18}
    
    feat = ['Fourier', 'Gauss']
    for row in range(2):
        axes[row,4].set_title('Exact kernel', **font)
        axes[row,4].set_xticks([])
        axes[row,4].set_yticks([])
        
        axes[row,0].set_ylabel('%s features' %feat[row], **font)
        for col, R in enumerate([1,10,100,1000]):
            axes[row,col].set_title(r'$\mathbf{Z} \mathbf{Z}^{\top}$, $R=%s$' % R, **font)
            axes[row,col].set_xticks([])
            axes[row,col].set_yticks([])
    
    # generate random 2D data
    N = 1000
    D = 2

    X = np.ones((N,D))
    X[:,0] = np.linspace(-3.,3.,N)
    X[:,1] = np.sort(np.random.randn(N))

    """ Start of your code 
    """
    for col, R in enumerate([1, 10, 100, 1000]):
        b = np.random.uniform(low=0, high=2 * np.pi, size=R)
        omega = np.random.normal(loc=0, scale=1, size=(D, R))
        K_app = np.zeros(shape=(N, N))

        for i in range(N):
            for j in range(N):
                z_x_transpose = (np.sqrt(2 / R) * np.cos(omega.T @ X[i] + b)).T
                z_x = np.sqrt(2 / R) * np.cos(omega.T @ X[j] + b)
                K_app[i, j] = z_x_transpose @ z_x
        axes[0, col].imshow(K_app)


    # computation of Gauss kernel
    K = np.zeros(shape=(N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = np.exp(- np.linalg.norm(X[i] - X[j]) ** 2 / 2)
    axes[0, 4].imshow(K)



    """ End of your code 
    """

    return fig

def task2():
    """ Subtask 2: Linear Regression with Feature Transforms

        Requirements for the plot:
        - the left and right subplots should cover the cases with random Fourier and Gauss features, respectively

        for each subplot:
        - plot the averaged (over 5 runs) mean and standard deviation of training and test errors over the number of features
        - include labels for the curves in a legend
    """

    def gen_data(n,d):
        sig = 1. 

        v_star = np.random.randn(d)
        v_star = v_star/np.sqrt((v_star**2).sum())

        # create input data on unit sphere
        x = np.random.randn(n,d)
        x = x/np.sqrt((x**2).sum(1,keepdims=True))
        
        # create targets y
        y = np.zeros((n))
        for n_idx in np.arange(n):
            y[n_idx] = 1/(0.25 + (x[n_idx]).sum()**2) + sig*np.random.randn(1)
        
        return x,y

    n = 200
    n_test = 100
    D = 5

    x_, y_ = gen_data(n+n_test,D)
    idx = np.random.permutation(np.arange(n+n_test))
    x,y,x_test,y_test = x_[idx][:n],y_[idx][:n],x_[idx][n::],y_[idx][n::]

    # features
    R = np.arange(1,100)

    # plot
    fig2, ax = plt.subplots(1,2)
    ax[0].set_title('Random Fourier Features')
    ax[0].set_xlabel('features R')

    ax[1].set_title('Random Gauss Features')
    ax[1].set_xlabel('features R')

    """ Start of your code 
    """



    """ End of your code 
    """

    ax[0].legend()
    ax[1].legend()

    return fig2

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    fig1 = task1()
    #fig2 = task2()
    pdf.savefig(fig1)
    #pdf.savefig(fig2)

    pdf.close()


