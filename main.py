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
    # Probability density functions with change of variables, 
    # check that you obtain a valid transformed pdf
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

        log_probs = np.zeros(shape=(K, S))
        x = np.tile(np.expand_dims(x, axis=1), (1, K, 1))
        diff = (x-mu)

        # We take the log, otherwise we encounter overflow/underflow error.
        for k in range(K):
            dk = diff[:,k,:].T
            tmp = np.linalg.inv(sigma[k] + 1e-6*np.eye(D)) @ dk
            log_exp = -0.5 * np.sum(dk * tmp, axis=0)

            choelsky = np.linalg.cholesky(sigma[k] + 1e-6 * np.eye(D)) + 1e-6 * np.eye(D)

            sigma_det = 2*np.sum(np.log(np.diagonal(choelsky)))  
            log_coef = -0.5 * (D * np.log(2 * np.pi) + sigma_det)            
            log_probs[k] = log_coef + log_exp
        
        return log_probs
    
    def em_algotithm(x, mu, sigma, pi):

        J = 15
        criteria = 0.01
        j = 0 
        converged = False
        likelihood = -np.inf

        while not converged:
            if j == J:
                converged = True

            else:
                j = j + 1
                # We calculate the logs of w's and then take the exponents.
                ws = np.zeros(shape=(K, S))
                log_probs = normal_dist(x, mu, sigma)
                log_pi = np.log(pi[:, np.newaxis])
                
                max_log_probs = np.max(log_probs, axis=0) # Maximum log prob of samples
                tmp = np.log(np.sum( pi[..., None] * (np.exp(log_probs - max_log_probs))
                                                                    , axis=0)) # log sum exp trick done

                ws = log_pi + log_probs - max_log_probs - tmp  
                ws = np.exp(ws)

                # We took this part from Assignment explanation.
                Nk = ws.sum(1) 
                mu = (ws[...,None]*x[None,...]).sum(1)/Nk[:,None] 
                diff = (x[None,...] - mu[:,None,:]) 
                sigma = np.einsum('KNi,KNj->Kij',ws[:,:,None]*diff,diff)/Nk[:,None,None]
                pi = Nk/S

                likelihood_next = np.sum(max_log_probs + tmp) 
                delta = np.abs(likelihood_next - likelihood)
                likelihood = likelihood_next
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
                cluster_info = {k:[] for k in range(K)}
                for data_point in x:
                    distance = np.sum((data_point - centroids)*(data_point - centroids), axis=1) 
                    prediction = np.argmin(distance)
                    cluster_info[prediction].append(data_point)

                new_centroids = np.zeros(centroids.shape)
                for cluster in cluster_info.keys():
                    new_centroids[cluster] = np.mean(cluster_info[cluster], axis=0)

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

    for i in range(num_samples):
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

    def normal_dist(x, mu, sigma):
        D = x.shape[1]

        log_probs = np.zeros(shape=(K, S))
        x = np.tile(np.expand_dims(x, axis=1), (1, K, 1))
        diff = (x - mu)

        for k in range(K):
            dk = diff[:, k, :].T
            tmp = np.linalg.inv(sigma[k] + 1e-6 * np.eye(D)) @ dk
            log_exp = -0.5 * np.sum(dk * tmp, axis=0)

            choelsky = np.linalg.cholesky(sigma[k] + 1e-6 * np.eye(D)) + 1e-6 * np.eye(D)
            sigma_det = 2 * np.sum(np.log(np.diagonal(choelsky)))
            log_coef = -0.5 * (D * np.log(2 * np.pi) + sigma_det)
            log_probs[k] = log_coef + log_exp

        return log_probs


    # Masking the samples
    mask_2d = np.reshape(mask, (sz, _))
    for i in range(S):
        x[i] = x[i] * mask_2d

    for s in range(S):
        ax[s, 0].imshow(x[s], vmin=0, vmax=1., cmap='gray')

    indices_of_one = np.flatnonzero(mask == 1)
    indices_of_zero = np.flatnonzero(mask == 0)
    length_x2 = len(indices_of_one)
    length_x1 = len(indices_of_zero)

    # Partition of the Mean
    mu_old = m_params[0] 
    mu_2 = mu_old[:, indices_of_one]
    mu_1 = mu_old[:, indices_of_zero]

    # Partition of the Covariance matrix
    sigma_old = m_params[1]
    sigma_12 = sigma_old[:, indices_of_zero][:, :, indices_of_one]
    sigma_22 = sigma_old[:, indices_of_one][:, :, indices_of_one]
    sigma_11 = sigma_old[:, indices_of_zero][:, :, indices_of_zero]
    sigma_21 = sigma_old[:, indices_of_one][:, :, indices_of_zero]

    # Partition of the variable
    x = np.reshape(x, [S, -1])
    x2 = np.zeros(shape=(S, len(indices_of_one))) 
    x1 = np.zeros(shape=(S, len(indices_of_zero)))
    x2[:, :] = x[:, indices_of_one]
    x1[:, :] = x[:, indices_of_zero]

    # Calculating the new Pi's.
    pi = m_params[2]
    probs = np.exp(normal_dist(x2, mu_2, sigma_22))
    numerator = pi[..., None] * probs
    denominator = np.sum(pi[..., None] * probs, axis = 0)[None, ...]
    pi_new = numerator / denominator

    # Calulating the posterior expecatation and plot.
    D = x2.shape[1]
    mu_1_2 = np.zeros(shape=(K, length_x1))
    inv_sigma_22 = np.linalg.inv(sigma_22 + 1e-6 * np.eye(D))
    tmp = sigma_12 @ inv_sigma_22

    for s in range(S):
        sample = x2[s]
        diff = sample[None, ...] - mu_2

        pi_1_2 = pi_new[:, s][..., None]
        mu_1_2 = mu_1 + np.squeeze((tmp @ diff[..., None]), axis=2)

        post_expectation = np.sum(pi_1_2 * mu_1_2, axis=0)

        reconstruct = np.zeros(shape=(sz**2))
        reconstruct[indices_of_one] = x2[s]
        reconstruct[indices_of_zero] = post_expectation
        reconstruct = reconstruct.reshape((sz,sz))

        ax[s, 1].imshow(reconstruct, cmap='gray')        

    
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
    size_of_img = 28*28
    int_value = int(size_of_img * 0.9)
    tmp_one = np.ones(size_of_img)
    tmp_zero_indices = np.random.choice(size_of_img, int_value, False)
    tmp_one[tmp_zero_indices] = 0
    mask = tmp_one

    fig2 = task3(x_test,mask,gmm_params)

    for f in fig1:
        pdf.savefig(f)
    pdf.savefig(fig2)
    pdf.close()
    