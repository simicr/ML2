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

    print('Computing approximation via Random Fourier...')
    for col, R in enumerate([1, 10, 100, 1000]):
        b = np.random.uniform(low=0, high=2 * np.pi, size=R)
        omega = np.random.normal(loc=0, scale=1, size=(D, R))

        Z = np.sqrt(2 / R) * np.cos(X @ omega + b)
        K_app = Z @ Z.T

        axes[0, col].imshow(K_app)

    print('Computing approximation via Random Gaus ...')
    for col, R in enumerate([1, 10, 100, 1000]):
        
        sigma = 1
        t = X[np.random.randint(low=0, high = N, size = R)]

        distances = np.linalg.norm(X[:, None, :] - t[None, :, :], axis=2) ** 2
        Z = np.sqrt(1 / R) * np.exp(-distances / (2 * sigma ** 2))
        K_app = Z @ Z.T

        axes[1, col].imshow(K_app)

        
    print('Computing exact kernels ...')
    c1 = 1
    c2 = 1
    sigma = 1
    
    pairwise_distances = np.linalg.norm(X[:, None, :] - X[:, None, :].transpose(1, 0, 2), axis=2)
    K1 = np.exp(-pairwise_distances ** 2 / 2)
    K2 = c1 * c2 * np.exp(-pairwise_distances ** 2 / (4 * sigma ** 2))

    axes[0, 4].imshow(K1)
    axes[1, 4].imshow(K2)
    print('Finished Task 1')

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
    def fourier(x, r):
        b = np.random.uniform(low=0, high=2 * np.pi, size=r)
        omega = np.random.normal(loc=0, scale=1, size=(D, r))
        
        return np.sqrt(2 / r) * np.cos(x @ omega + b)

    def gaus(x, n, r):
        sigma = 1
        t = x[np.random.randint(low=0, high = n, size = r)]
        distances = np.linalg.norm(x[:, None, :] - t[None, :, :], axis=2) ** 2

        return np.sqrt(1 / r) * np.exp(-distances / (2 * sigma ** 2))
    
    def train_and_test(PHI, PHI_test, lmd):

        train_loss = 0
        test_loss = 0

        theta = np.linalg.inv(PHI.T @ PHI + lmd*np.eye(r)) @ ( PHI.T @ y) # What to do if singular?
        train_prediction = PHI @ theta[..., None]
        test_prediction = PHI_test @ theta[..., None]
        
        train_loss = (np.sum((y[..., None] - train_prediction) ** 2) / n)
        test_loss = (np.sum((y_test[..., None] - test_prediction) ** 2) / n_test)
        
        return train_loss, test_loss

    experiments = 5
    sigma = 1.0
    c1 = 1.0
    c2 = 1.0

    lamf = 1
    lamg = 1    
    fourier_history = np.zeros(shape=(2, R.shape[0], experiments))
    gaus_history = np.zeros(shape=(2, R.shape[0], experiments)) 


    for e in range(experiments):
        
        for r in R:

            # Feature transformation
            PHIF_train = fourier(x, r)
            PHIF_test = fourier(x_test, r)

            PHIG_train = gaus(x, n, r) 
            PHIG_test = gaus(x_test, n_test, r)

            # Train and test
            f_train , f_test = train_and_test(PHIF_train, PHIF_test, lamf)
            g_train , g_test = train_and_test(PHIG_train, PHIG_test, lamg)

            # Appending the results
            fourier_history[0, r - 1, e] = f_train
            fourier_history[1, r - 1, e] = f_test
            gaus_history[0, r - 1, e] = g_train
            gaus_history[1, r - 1, e] = g_test

    for i, history in enumerate([fourier_history, gaus_history]):

        train_loss_mean = np.mean(history[0], axis=1)
        test_loss_mean = np.mean(history[1], axis=1)
        train_loss_std = np.std(history[1], axis=1)
        test_loss_std = np.std(history[1], axis=1)

        ax[i].plot(R, train_loss_mean, label='Train Loss')
        ax[i].fill_between(R, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, alpha=0.3)
        ax[i].plot(R, test_loss_mean, label='Test Loss')
        ax[i].fill_between(R, test_loss_mean - test_loss_std, test_loss_mean + test_loss_std, alpha=0.3)


    # Needs to be reviewed.

    def train_and_evaluate_kernal(Ktrain, Ktest, lmd):

        a = -np.linalg.inv(Ktrain + lmd*np.eye(n)) @ (lmd*np.eye(n) @ y)

        train_prediction = (-1/lmd) * Ktrain @ a
        test_prediction = (-1/lmd) * Ktest @ a

        train_loss = (np.sum((y - train_prediction) ** 2) / n)
        test_loss = (np.sum((y_test - test_prediction) ** 2) / n_test)
        
        return train_loss, test_loss
    
    # Task 3 -> calculating the kernels
    pairwise_distances = np.linalg.norm(x[:, None, :] - x[:, None, :].transpose(1, 0, 2), axis=2)
    pairwise_distances2 = np.linalg.norm(x_test[:, None, :] - x[:, None, :].transpose(1, 0, 2), axis=2)

    KF_train = np.exp(-pairwise_distances ** 2 / 2)
    KF_test = np.exp(-pairwise_distances2 ** 2 / 2)

    KG_train = c1 * c2 * np.exp(-pairwise_distances ** 2 / (4 * sigma ** 2))
    KG_test = c1 * c2 * np.exp(-pairwise_distances2 ** 2 / (4 * sigma ** 2))

    # Task 3 -> evaluating the loss
    kf_train_loss, kf_test_loss = train_and_evaluate_kernal(KF_train, KF_test, lamf)
    kg_train_loss, kg_test_loss = train_and_evaluate_kernal(KG_train, KG_test, lamg)

    print('KF', kf_train_loss, kf_test_loss)
    print('KG', kg_train_loss, kg_test_loss)

    # Task 3 -> plotting
    for i, (train, test) in enumerate([(kf_train_loss, kf_test_loss),(kg_train_loss, kg_test_loss)]):
        ax[i].axhline(y=train, color = 'blue', label='Kernal train', linestyle='--')
        ax[i].axhline(y=test,color ='orange', label='Kernal test', linestyle='--')
        
    """ End of your code 
    """

    ax[0].legend()
    ax[1].legend()

    return fig2

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    fig1 = task1()
    fig2 = task2()

    pdf.savefig(fig1)
    pdf.savefig(fig2)

    pdf.close()


