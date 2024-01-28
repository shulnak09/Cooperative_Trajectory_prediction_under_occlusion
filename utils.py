
import numpy as np
from matplotlib import pyplot as plt
import numpy as np




def average_displacement_error(mean, ground_truth):
    
    ens_ADE = []
    for n in range(mean.shape[0]):
        ADE = np.sqrt(np.sum((np.squeeze(mean[n,:,:,:2])  - np.squeeze(ground_truth[n, 8:,:2]))**2, axis =1))
        ens_ADE.append(np.mean(ADE))   
    return np.array(ens_ADE)
        

def final_displacement_error(mean, ground_truth):
    ens_FDE = []
    for n in range(mean.shape[0]):
        FDE = np.sqrt(np.sum((np.squeeze(mean[n,:,-1,:2])  - np.squeeze(ground_truth[n, -1,:2]))**2))
        ens_FDE.append(FDE)
    return np.array(ens_FDE)

def relative_error(trajectory_1, trajectory_2):
    '''
    Computes the relative estimation error between actual (ego agent) and estimated trajectory 
    (static camera transformed) obtained using pose recovery [R|t]
    '''
    for i in range(trajectory_1.shape[0]):
        rel_error = np.sqrt(np.sum((np.squeeze(trajectory_1[8:,:2])  - np.squeeze(trajectory_2[8:,:2]))**2, axis =1))
    return rel_error


def KL_div(mu_1, mu_2,  sigma_1, sigma_2):
    '''
    Computes the KL-divergence between the posterior predictive distribution 
    for Static camera transformed and ego agent.
    '''
    
    n = 2 # Bivariate Normal
    
    mahalanobis_dist = [(mu_2 -mu_1).T @ np.linalg.inv(sigma_2) @ (mu_2 -mu_1)]
    shape_factor = np.trace(np.linalg.inv(sigma_2) @ sigma_1)
    negative_log = np.log(np.abs(sigma_1)/np.abs(sigma_2))
    n = 2 # For a bivariate normal distribution
    
    KL_div =  0.5*[mahalanobis_dist + shape_factor - negative_log -n]
    return KL_div
    

def PICP():
    '''
    Computes the Predictive interval covergae peobability:
    Percentage of ground truth that lies within predictive uncertainty at each state
    '''
    pass

def MPIW():
    '''
    Compute the Mean Predictive Interval Width:
    Average width of the predictive covariance ellipse.
    '''
    pass