
import numpy as np
from matplotlib import pyplot as plt
import numpy as np




def average_displacement_error(mean):
    ADE = 0
    ADE += np.sqrt(np.sum((np.squeeze(mean[1,:,:,:2])  - np.squeeze(mean[2,:, :,:2]))**2, axis =1))
    return np.mean(ADE)
        

def final_displacement_error(mean):
    FDE = np.sqrt(np.sum((np.squeeze(mean[1,:,-1,:2])  - np.squeeze(mean[2,:, -1,:2]))**2))
    return FDE

def relative_error(trajectory_1, trajectory_2):
    '''
    Computes the relative estimation error between actual (ego agent) and estimated trajectory 
    (static camera transformed) obtained using pose recovery [R|t]
    '''
    rel_error = 0
    rel_error += np.sqrt(np.sum((np.squeeze(trajectory_1[:,:2])  - np.squeeze(trajectory_2[:,:2]))**2, axis =1))
    return np.mean(rel_error)


def KL_divergence(mu_1, mu_2,  sigma_1, sigma_2):
    '''
    Computes the KL-divergence between the posterior predictive distribution 
    for Static camera transformed and ego agent.
    '''
    mu_1 = mu_1.reshape(-1, 1)
    mu_2 = mu_2.reshape(-1, 1)
    
    mahalanobis_dist = ((mu_2-mu_1).T @ np.linalg.inv(sigma_2) @ (mu_2-mu_1))
    shape_factor = np.trace(np.linalg.inv(sigma_2) @ sigma_1)
    negative_log = np.log(np.linalg.det(sigma_1)/np.linalg.det(sigma_2))
    n = 2 # For a bivariate normal distribution
    KL_div =  0.5*(mahalanobis_dist + shape_factor - negative_log -n)
    return KL_div

def entropy(sigma):
    det_sigma = np.linalg.det(sigma)
    return 0.5 * np.log((2 * np.pi * np.e)**2 * det_sigma)
    

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