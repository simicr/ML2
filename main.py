""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from matplotlib.backends.backend_pdf import PdfPages

def get_sigmas(sigma_1, sigma_L, L):
    # geometric progression for noise levels from \sigma_1 to \sigma_L 
    return torch.tensor(np.exp(np.linspace(np.log(sigma_1),np.log(sigma_L), L)))

def generate_data(n_samples):
    """ Generate data from 3-component GMM

        Requirements for the plot: 
        fig1 
            - this plot should contain a 2d histogram of the generated data samples

    """
    x, mu, sig, a = torch.zeros((n_samples,2)), None, None, None
    fig1 = plt.figure(figsize=(5,5))
    plt.title('Data samples')

    """ Start of your code
    """

    print('Generating data ...')
    mu = 0.25 * np.array([[1, 1], [3, 1], [2, 3]])
    sig = np.array([np.diag([0.01, 0.01]), np.diag([0.01, 0.01]), np.diag([0.01, 0.01])])
    a = np.array([1/3, 1/3, 1/3])

    for e in range(n_samples):
        k = np.random.choice(a.shape[0], p=a)
        x[e] = torch.tensor(np.random.multivariate_normal(mu[k], sig[k]))
    
    plt.hist2d(x[:, 0].numpy(), x[:, 1].numpy(), bins=128)

    """ End of your code
    """

    return x, (mu, sig, a), fig1

def dsm(x, params):
    """ Denoising score matching
    
        Requirements for the plots:
        fig2
            - ax2[0] contains the histogram of the data samples
            - ax2[1] contains the histogram of the data samples perturbed with \sigma_1
            - ax2[2] contains the histogram of the data samples perturbed with \sigma_L
        fig3
            - this plot contains the log-loss over the training iterations
        fig4
            - ax4[0,0] contains the analytic density for the data samples perturbed with \sigma_1
            - ax4[0,1] contains the analytic density for the data samples perturbed with an intermediate \sigma_i 
            - ax4[0,2] contains the analytic density for the data samples perturbed with \sigma_L

            - ax4[1,0] contains the analytic scores for the data samples perturbed with \sigma_1
            - ax4[1,1] contains the analytic scores for the data samples perturbed with an intermediate \sigma_i 
            - ax4[1,2] contains the analytic scores for the data samples perturbed with \sigma_L
        fig5
            - ax5[0,0] contains the learned density for the data samples perturbed with \sigma_1
            - ax5[0,1] contains the learned density for the data samples perturbed with an intermediate \sigma_i 
            - ax5[0,2] contains the learned density for the data samples perturbed with \sigma_L

            - ax5[1,0] contains the learned scores for the data samples perturbed with \sigma_1
            - ax5[1,1] contains the learned scores for the data samples perturbed with an intermediate \sigma_i 
            - ax5[1,2] contains the learned scores for the data samples perturbed with \sigma_L
    """
    
    fig2, ax2 = plt.subplots(1,3,figsize=(10,3))
    ax2[0].hist2d(x.cpu().numpy()[:,0],x.cpu().numpy()[:,1],128), ax2[0].set_title(r'data $x$')
    ax2[1].set_title(r'data $x$ with $\sigma_{1}$')
    ax2[2].set_title(r'data $x$ with $\sigma_{L}$')

    fig3, ax3 = plt.subplots(1,1,figsize=(5,3))
    ax3.set_title('Log loss over training iterations')

    # plot analytic density/scores (fig4) vs. learned by network (fig5)
    fig4, ax4 = plt.subplots(2,3,figsize=(16,10))
    fig5, ax5 = plt.subplots(2,3,figsize=(16,10))

    mu, sig, a = params

    """ Start of your code
    """

    # Class of the neural network. 
    class Network(torch.nn.Module):

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.fc1 = nn.Linear(in_features=3, out_features=128, bias=True, dtype=torch.float64)
            self.fc2 = nn.Linear(in_features=128, out_features=32, bias=True, dtype=torch.float64)
            self.fc3 = nn.Linear(in_features=32, out_features=8, bias=True, dtype=torch.float64)
            self.fc4 = nn.Linear(in_features=8, out_features=1, bias=True, dtype=torch.float64)

        def count_parameters(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        def forward(self, x):
            x = F.elu(self.fc1(x))
            x = F.elu(self.fc2(x))
            x = F.elu(self.fc3(x))
            return self.fc4(x)

    # Compute the density under each component.
    def prob_per_component(points, mu, sig): 
        
        D = points.shape[2]
        K = mu.shape[0]
        prob = np.zeros((K, points.shape[0], points.shape[1]))
        
        for k in range(K):
            norm_const = 1 / (np.sqrt((2 * np.pi) ** D * np.linalg.det(sig[k])))
            inv_cov = np.linalg.inv(sig[k])
            diff = points - mu[k]

            exponent = np.einsum('...k,kl,...l->...', diff, inv_cov, diff)
            prob[k] = norm_const * np.exp(-0.5 * exponent)

        return prob

    # Compute analyticly the density of the GMM.
    def gmm_density(points, mu, sig, a):

        probs = prob_per_component(points, mu, sig)
        return np.sum(a[:, None, None] * probs, axis=0)
    
    # Analyticly derived the score function of the GMM. 
    def gmm_score(points, mu, sigma, a):
        
        K = mu.shape[0]

        probs = prob_per_component(points, mu, sigma) 
        gmmdensity = gmm_density(points, mu, sigma, a) 
        
        exponent_gradient = np.zeros((K, points.shape[0], points.shape[1], points.shape[2])) 
        
        for k in range(K):
            diff = points - mu[k]
            inv_cov = np.linalg.inv(sig[k])
            exponent_gradient[k] = np.einsum('kl,...l->...k', inv_cov, diff)

        normal_gradient = probs[..., None] * exponent_gradient 
        gmm_gradient = np.sum(a[:, None, None, None] * normal_gradient, axis=0)

        return -0.5* gmm_gradient / gmmdensity[..., None]

    sigma_1 , sigma_L, L = 0.05 , 1.0, 100

    noiselevels = [sigma_1, 0.1 ,sigma_L]
    Net = Network()
    sigmas_all = get_sigmas(sigma_1=sigma_1, sigma_L=sigma_L, L=L)


    print('Visualization of the pertrub data ...')
    n_samples = x.size(0)
    for idx, noise in enumerate([0, sigma_1, sigma_L]):
        z = torch.tensor(np.random.multivariate_normal(mean=[0,0], cov=np.eye(2), size=n_samples))
        bar_x = x + noise * z
        ax2[idx].hist2d(bar_x[:, 0].numpy(), bar_x[:, 1].numpy(), bins=128)
    
    
    for param in Net.parameters():
        param.requires_grad = True
    x = x.double()
    x.requires_grad = True

    print(f'Training the model with {Net.count_parameters()} parameters...')

    epochs = 600
    optimer = optim.Adam(lr=1e-2, params=Net.parameters())
    criterion = nn.MSELoss()
    history = []  

    freq = 20
    for e in range(epochs):

        random_indices = torch.randint(0, L, size=(n_samples, 1))
        sigmas = sigmas_all[random_indices]
        z = torch.tensor(np.random.multivariate_normal(mean=[0,0], cov=np.eye(2), size=n_samples))

        bar_x = x + sigmas*z
        x_with_sigma = torch.cat([bar_x, sigmas], dim=1)


        optimer.zero_grad()
        
        predicted_energy = Net(x_with_sigma)
        predicted = -sigmas * torch.autograd.grad(
            outputs=[predicted_energy[i] for i in range(n_samples)],
            inputs=x_with_sigma, create_graph=True)[0][:, 0:2]
        
        target = (x - bar_x) / sigmas
        
        loss = criterion(predicted, target)
        
        loss.backward()
        optimer.step()

        history.append(loss.item())
        if (e % freq == 0) or ((e + 1) % epochs == 0):
            print(f'\t Epoch: {e} | Loss: {loss.item()}')

    ax3.plot(history)
    ax3.set_yscale('log')



    print('Analytic density and score ...')
    X = np.linspace(-0.8, 1.8, 50)
    Y = np.linspace(1.8, -0.8, 50)
    XM, YM = np.meshgrid(X, Y)
    points = np.dstack((XM, YM))

    for idx, noiselevel in enumerate(noiselevels):
        ax4[0, idx].imshow(gmm_density(points, mu, sig + ( noiselevel ** 2 ) * np.eye(2), a))
        score = gmm_score(points, mu, sig + ( noiselevel ** 2 ) * np.eye(2), a)
        ax4[1, idx].quiver(X, Y, score[:, :, 0], score[:, :, 1], np.hypot(score[:, :, 0], score[:, :, 1]))
        
    print('Learned density and score ...')
    for idx, noiselevel in enumerate(noiselevels):
        noise = torch.full(size=(50, 50, 1), fill_value=noiselevel, dtype=torch.float64)
        points_with_noise = torch.cat((torch.tensor(points, dtype=torch.float64), noise), dim=2)
        points_with_noise.requires_grad_(True)

        predicted_energy = Net(points_with_noise)
        exp_predicted_energy = torch.exp(-predicted_energy)
        ax5[0, idx].imshow(exp_predicted_energy.detach().numpy())

        outputs = []
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                outputs.append(predicted_energy[i,j])

        score = -torch.autograd.grad(
                outputs=outputs, 
                inputs=points_with_noise
            )[0][:, :, 0:2]
        
        score_x_dim = score[:, :, 0].detach().numpy()
        score_y_dim = score[:, :, 1].detach().numpy()
        ax5[1, idx].quiver(X, Y, score_x_dim, score_y_dim, np.hypot(score_x_dim, score_y_dim))

    """ End of your code
    """

    for idx, noiselevel in enumerate(noiselevels):
        ax4[0,idx].set_title(r'$\sigma$=%f' %noiselevel)
        ax5[0,idx].set_title(r'$\sigma$=%f' %noiselevel)

        ax4[0,idx].set_xticks([]), ax4[0,idx].set_yticks([])
        ax4[1,idx].set_xticks([]), ax4[1,idx].set_yticks([])
        ax5[0,idx].set_xticks([]), ax5[0,idx].set_yticks([])
        ax5[1,idx].set_xticks([]), ax5[1,idx].set_yticks([])

    ax4[0,0].set_ylabel('analytic density'), ax4[1,0].set_ylabel('analytic scores')
    ax5[0,0].set_ylabel('learned density'), ax5[1,0].set_ylabel('learned scores')

    return Net, sigmas_all, (fig2, fig3, fig4, fig5)

def sampling(Net, sigmas_all, n_samples):
    """ Sampling from the learned distribution
    
        Requirements for the plots:
            fig6
                - ax6[0] contains the histogram of the data samples
                - ax6[1] contains the histogram of the generated samples
    
    """
    
    fig6, ax6 = plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    ax6[0].set_title(r'data $x$')
    ax6[1].set_title(r'samples')

    """ Start of your code
    """
    x, (mu, sig, a), fig1 = generate_data(n_samples)
    ax6[0].hist2d(x[:, 0].numpy(), x[:, 1].numpy(), bins=128)

    eps = 0.01
    T = 20
    L = sigmas_all.size(0)
    sample = torch.tensor(np.random.multivariate_normal(mean=[0,0], cov=np.eye(2), size=n_samples), requires_grad=True)
    
    for i in range(L-1, 0, -1):
        sigma = sigmas_all[i]
        alpha_i = eps * (sigma ** 2 / sigmas_all[0] ** 2)

        for _ in range(T):
            
            noise = torch.full((n_samples, 1), sigma)
            sample_w_noise = torch.cat([sample, noise], dim=1)
            output_network = Net(sample_w_noise)
            grad_f = torch.autograd.grad(outputs=[output_network[i] for i in range(n_samples)], inputs=sample_w_noise, allow_unused=True)[0][:, 0:2]

            z = torch.sqrt(alpha_i) * torch.tensor(np.random.multivariate_normal(mean=[0,0], cov=np.eye(2), size=n_samples))

            sample = sample - (alpha_i / 2) * grad_f + z

    
    ax6[1].hist2d(sample[:, 0].detach().numpy(), sample[:, 1].detach().numpy(), bins=128)
    print('Finished!')
    """ End of your code
    """

    return fig6 

if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')

    # generate data
    x, params, fig1 = generate_data(n_samples=10000)

    # denoising score matching
    Net, sigmas_all, figs = dsm(x=x, params=params)

    # sampling
    fig6 = sampling(Net=Net, sigmas_all=sigmas_all, n_samples=5000)

    pdf.savefig(fig1)
    for f in figs:
        pdf.savefig(f)
    pdf.savefig(fig6)

    pdf.close()
    
