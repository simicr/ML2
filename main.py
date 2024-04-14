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

from scipy.integrate import quad

def task1():
    # probability density functions with change of variables, check that you obtain a valid transformed pdf
    
    """ Start of your code
    """
    def pz(z): 
        if z < 0:
            return 0
        
        return np.exp(-0.5*z) / (np.sqrt(2*np.pi*z)) 
    
    print('Approximation of the integral and the absolute err: ', quad(pz, -np.inf, np.inf) )
    
    """ End of your code
    """

def task2(x, K):
    """ Multivariate GMM

        Requirements for the plots: 
        fig1
            - ax[0,k] plot the mean of each k GMM component, the subtitle contains the weight of each GMM component
            - ax[1,k] plot the covariance of each k GMM component
        fig2 
            - ax[k,0] plot the selected *first* reshaped line of the k-th covariance matrix
            - ax[k,1] plot the selected *second* reshaped line of the k-th covariance matrix
            - ax[k,2] plot the selected *third* reshaped line of the k-th covariance matrix
            - ax[k,3] plot the selected *fourth* reshaped line of the k-th covariance matrix
        fig3: 
            - plot the 8 samples that were sampled from the fitted GMM
    """
    
    mu, sigma, pi = [], [], np.zeros((K)) # modify this later
    num_samples = 10

    fig1, ax1 = plt.subplots(2, K, figsize=(2*K,4))
    fig1.suptitle('Task 2 - GMM components', fontsize=16)

    fig2, ax2 = plt.subplots(2, num_samples//2, figsize=(2*num_samples//2,4))
    fig2.suptitle('Task 2 - samples', fontsize=16)

    """ Start of your code
    """

    def normal_dist(x, mu, sigma):
        D = x.shape[1]

        probs = np.zeros(shape=(K, S))
        
        x = np.tile(np.expand_dims(x, axis=(K-1)), (1, 1, K)) # (S, D) -> (S, D, K)  
        mu = mu.T # (K, D) -> (D, K) 
        diff = (x-mu) # (S,D,K) 

        for k in range(K):
            dk = diff[:,:,k]
            tmp = np.einsum('ij,Nj->Ni',np.linalg.inv(sigma[k] + 1e-6*np.eye(D)),dk) 
            tmp = np.einsum('Ni,Ni->N',dk,tmp)
            probs[k] = -0.5*tmp

        coef = 0.005 # Needs to be changed later.
        
        return  coef*np.exp(probs) #(K, S)
    
    def em_algotithm(x, mu, sigma, pi):

        J = 25
        criteria = 0.1
        j = 0 
        converged = False
        likelihood = -np.inf

        while not converged:
            if j == J:
                converged = True
            else:
                j = j + 1

                ws = np.zeros(shape=(K, S))
                probs = normal_dist(x, mu, sigma)

                # Could be changed 
                for k in range(K):
                    weighted_probs = pi[k]*probs[k]
                    ws[k] = weighted_probs/np.sum(weighted_probs)
                ws = ws[:,:,np.newaxis]

                for k in range(K):
                    Nk = np.sum(ws[k])
                    mu[k] = np.sum( ws[k]*x, axis=0)/np.sum(Nk)
                    outer_product = np.einsum('ij,ik->ijk', (x-mu[k]), (x-mu[k]))
                    sigma[k] = sigma[k] # Think of a way given the outer product
                    pi[k] = Nk/pi[k]


                likelihood_next = 0 
                delta = np.abs(likelihood_next - likelihood)
                likelihood  = likelihood_next
                converged = delta < criteria
        
        print('EM algorithm finished in ', j, ' iteration.')
        return mu, sigma, pi
    
    def kmeans_algorithm(x):
        j = 0
        J = 25 
        converged = False
        criteria = 0.005

        centroids = x[:K] 
        while not converged:

            if j == J:
                converged = True
            else:
                j = j + 1
                cluster_info = {k:[] for k in range(K)} # Could maybe do something different?
                for data_point in x:
                    distance = np.sum((data_point - centroids)*(data_point - centroids), axis=1) 
                    prediction = np.argmin(distance)
                    cluster_info[prediction].append(data_point)

                new_centroids = np.zeros(centroids.shape)
                for cluster in cluster_info.keys():
                    new_centroids[cluster] = np.mean(cluster_info[cluster], axis=0) # Good shape but are the values good?

                
                delta = np.max(np.abs(new_centroids - centroids))
                converged = delta < criteria
                centroids = new_centroids

        print('K-means algorithm finished in ', j, ' iteration.')
        return centroids

    S = x.shape[0]
    M = x.shape[1]
    x = np.reshape(x, [2000, -1])


    mu = kmeans_algorithm(x)
    for k in range(K):
        sigma.append(np.identity(M*M))
    pi = np.ones(K)/K
    mu, sigma, pi = em_algotithm(x, mu, sigma, pi)
    


    for k in range(K):
        ax1[0,k].imshow(mu[k].reshape(M, M), cmap='gray')
        ax1[1,k].imshow(sigma[k], cmap='viridis')

    for i in range(10):
        k = np.random.choice(K, p=pi)
        sample = np.random.multivariate_normal(mu[k], sigma[k])
        ax2[i // 5, i % 5].imshow(sample.reshape(M,M), cmap='gray')

    
    """ End of your code
    """

    for k in range(K):
        ax1[0,k].set_title('C%i with %.2f' %(k,pi[k])), ax1[0,k].axis('off'), ax1[1,k].axis('off')

    return (mu, sigma, pi), (fig1,fig2)

def task3(x, mask, m_params):
    """ Conditional GMM

        Requirements for the plots: 
        fig
            - ax[s,0] plot the corrupted test sample s
            - ax[s,1] plot the restored test sample s (by using the posterior expectation)
            - ax[s,2] plot the groundtruth test sample s 
    """
    
    S, sz, _ = x.shape

    fig, ax = plt.subplots(S,3,figsize=(3,8))
    fig.suptitle('Task 3 - Conditional GMM', fontsize=12)
    for a in ax.reshape(-1):
        a.axis('off')
        
    ax[0,0].set_title('Condition',fontsize=8), ax[0,1].set_title('Posterior Exp.',fontsize=8), ax[0,2].set_title('Groundtruth',fontsize=8)
    for s in range(S):
        ax[s,2].imshow(x[s], vmin=0, vmax=1., cmap='gray')

    """ Start of your code
    """



    """ End of your code
    """

    return fig

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    # Task 1: transformations of pdfs
    task1()

    # load train and test data
    with np.load("data.npz") as f:
        x_train = f["train_data"]
        x_test = f["test_data"]

    # Task 2: fit GMM to FashionMNIST subset
    K = 3 # TODO: adapt the number of GMM components
    gmm_params, fig1 = task2(x_train,K)

    # Task 2: inpainting with conditional GMM
    mask = None
    fig2 = task3(x_test,mask,gmm_params)

    for f in fig1:
        pdf.savefig(f)
    pdf.savefig(fig2)
    pdf.close()
    