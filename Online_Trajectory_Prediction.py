import numpy as np
import random
from scipy.linalg import inv
import pickle as pkl
import seaborn as sns
from matplotlib import pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import os
from tqdm import trange
import torch.optim as optim
import matplotlib.font_manager
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from scipy.spatial.transform import Rotation
import pdb

# Import classes
from kalman_module import *
from model import lstm_encoder, lstm_decoder, lstm_seq2seq
from Cooperative_tracking import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def error_covariance_ellipse(X_test, y_test, mus, sigmas, ground_cov, ax,  id_no =100):
    '''
    This function predicts the 12 future states of the pedestrian provided 8 historical steps.
    It also plots the 2-sigma covariance ellipse around each state highlighting the
    uncertainty bound. 
    '''
    X_test, y_test = np.expand_dims(X_test, axis = 0), np.expand_dims(y_test, axis=0)
    sigmas, mus = np.expand_dims(sigmas, axis = 0), np.expand_dims(mus, axis = 0)
    
    
    # num_fea = mus.shape[3]
    sigmas = np.exp(sigmas)
    mu_ens = np.mean(mus, axis=0)

    var_aleatoric = np.mean(sigmas[:,:,:,:2], axis = 0)
    var_epistemic = np.mean(mus[:,:,:,:2]**2, axis = 0) - mu_ens[:,:,:2]**2
    var_ens = var_aleatoric  + var_epistemic
    var_state_unc = (np.mean((mus[:,:,:,2:4]), axis=0)) 
    ground_cov = ground_cov.transpose(1,0,2,3)


    ax.scatter(X_test[id_no,:,0],  X_test[id_no,:,1], color='g',marker='o',s =15, zorder=1) # Input  
    ax.scatter(y_test[id_no,:,0],  y_test[id_no,:,1], color='r', marker='^', s = 15, zorder=2)
    ax.plot(mu_ens[id_no,:,0], mu_ens[id_no,:,1], color='b', marker='d', alpha=0.85, ms = 5, label = 'NN state Estimate', zorder =3 )

   
    
    state_cov = []
    pred_cov = []
    mu = []
    forward_pred = y_test.shape[1]
    for pred in range(forward_pred): 
        
        mean = np.squeeze(mu_ens[id_no, pred, :])
        
        # Predictive Variance:
        cov_pred = np.squeeze(np.squeeze(np.diag(var_ens[id_no,pred,:]))) 
        lambda_tot, v_tot = np.linalg.eig(cov_pred)
        lambda_tot = np.sqrt(lambda_tot)


        # State Variance:
        cov_state = np.squeeze(var_state_unc[id_no, pred,:2])
        cov_state = np.diag(np.squeeze(cov_state))
        lambda_ale, v_ale = np.linalg.eig(cov_state)
        lambda_ale = np.sqrt(lambda_ale)

        
        for j in range(2,3):
            ell1 = Ellipse(xy = (mean[0], mean[1]),
                     width = ( 1* lambda_ale[0] + lambda_tot[0]) * j* 2 ,
                     height = ( 1*lambda_ale[1] + lambda_tot[1]) *j* 2,
                        angle = np.rad2deg(np.arccos((v_ale[0,0]))),
                         color = 'none',  lw = 0.5) 
            ell1.set_facecolor('blue')
            ell1.set_alpha(0.1/j)
            # ax.add_artist(ell1)
            
            ell2 = Ellipse(xy = (mean[0], mean[1]),
                 width = (lambda_tot[0]) * j* 2,
                 height = (lambda_tot[1]) *j* 2,
                    angle = np.rad2deg(np.arccos((v_ale[0,0]))),
                 color = 'none', linestyle  ='--', lw = 0.25)
            ell2.set_facecolor('tab:olive')
            ell2.set_alpha(0.25/j)
            ax.add_artist(ell2)
            
            
        state_cov.append(cov_state)
        pred_cov.append(cov_pred)
        mu.append(mean)
    state_cov = np.array(state_cov)
    pred_cov = np.array(pred_cov)
    mu = np.array(mu)

    ax.set_aspect('equal', adjustable='box')
    plt.rcParams.update({'font.size': 14})
    
    ax.set_xlim([-6,6])
    ax.set_xticks([-2,0,2])
    ax.set_ylim([-2,14])
    csfont = {'fontname':'P052'}
#     hfont = {'fontname':'Nimbus Sans'}
    ell1.set_label("NN State Uncertainty $(2\sigma)$")
    ell2.set_label("NN Prediction Uncertainty $(\sigma$)")
    ax.set_ylabel('y (m)', **csfont,fontsize=16, fontweight ='normal')
    # ax.grid()
    # ax.legend(loc = 'lower right')
    ax.set_xlabel('x (m)', **csfont,fontsize=16, fontweight ='normal')

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
        label.set_fontweight('normal')

    plt.grid("on", alpha = 0.25)
    plt.setp(ax.get_xticklabels(), fontsize=16)




# # Run the cooperative Tracking ONLINE  using cameras:
# traj_folder = 'orientation_8'
# [traj_1, traj_1_transf, traj_2] = cooperative_estimation(traj_folder = traj_folder)
# online_data = [traj_1, traj_1_transf, traj_2]


# Below we compute the performance metrics OFFLINE:
# Open the saved .pkl files for obtaining performance metrics:
traj_folder = './frames_goodwin/TIV_results/Results/Orientation_8'

# Define the file names:
file_names = ['Static_cam.pkl', 'Static_cam_transf.pkl', 'Ego_agent.pkl']

online_data = []
for file_name in file_names:
    file_path = os.path.join(traj_folder, file_name )
    
    
    with open(file_path, 'rb') as file:
        online_data.append(pkl.load(file))

# Start time:
start_time = time.time()


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7.5,3), sharex=False, sharey=False)

sigma_preds = []
mu_preds = []


for plot_id in range (len(online_data)):
    train_traj = np.expand_dims(online_data[plot_id], axis =0)
    train_traj = train_traj - train_traj[:,0,:]
    # print("train_traj : {}".format(train_traj))
    # X_train, y_train = np.split(train_traj, [8,12], axis = 1)


    # Sampled Trajectory from KF posterior distribution: 
    num_traj = 1 # Number of Sampled Trajectories
    batch_traj_test, test_mu, test_cov =[], [], []
    for id in range(train_traj.shape[0]):
        X_seq = train_traj[id,:,:]
        trajectories, mus, covs = get_N_trajectories(X_seq, num_traj)
        batch_traj_test.append(trajectories)
        test_mu.append(mus)
        test_cov.append(covs)

    _, test_mu, test_cov = np.array(batch_traj_test), np.array(test_mu), np.array(test_cov)
    # print("Batch_Traj :", batch_traj)
    # print(batch_traj.shape)


    train_traj = np.expand_dims(train_traj, axis = 0)
    num_fea = 2 
    batch_traj = torch.tensor(train_traj).to(device)
    X_train_KF, y_train_KF = torch.split(batch_traj[:,:,:,:num_fea],[8, 12], dim = 2)
    batch_gaussian = np.concatenate([batch_traj[:,:,:,:num_fea], test_cov[:,:,:,:num_fea]], axis = 3)
    batch_gaussian = torch.tensor(batch_gaussian).float().to(device)
    batch_gaussian_input, batch_gaussian_output = torch.split(batch_gaussian, [8,12], dim =2) 
    
    # Load the MCD/DE saved NN model weights and biases as .pt file for inference
    PATH = './MCD_models/lstm_seq2seq_eth_zara01_zara02.pt'
    num_fea = batch_gaussian_input.shape[3] # Number of states
    model = lstm_seq2seq(input_size=num_fea, hidden_size =128).to(device) 
    # model.load_state_dict(torch.load(PATH))
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval() # Model set to evaluation set

    forward_pred = 12
    look_back = 8
    min_logvar, max_logvar = -4, 4


    with torch.no_grad():
        # for i in random.sample(idx, Num_ens):
        y_train_pred = model.predict(batch_gaussian_input[:,0,:,:], forward_pred, device)
        # print(y_train_pred.shape)
        y_train_pred_mu,   y_train_state_logvar, y_train_pred_logvar = y_train_pred[:,:,:int(num_fea/2)], (y_train_pred[:,:,int(num_fea/2):num_fea]), (y_train_pred[:,:,num_fea:])#target_len, b, 8
        y_train_state_var = torch.exp(y_train_state_logvar)
        y_train_pred_logvar = torch.clamp(y_train_pred_logvar, min=min_logvar, max=max_logvar)
        y_train_pred_mean = torch.cat((y_train_pred_mu, y_train_state_var),2)
        mse_train = ((y_train_pred_mean - batch_gaussian_output[:,0,:,:num_fea])**2).mean()
        #  print(f"Train MSE: {mse_train}")


    mu_preds.append(y_train_pred_mean)
    sigma_preds.append(y_train_pred_logvar)  
    

    # Compute the time it takes to predict Trajectory:
    end_time = time.time()
    elapsed_time =  end_time - start_time
    print("Total Elapsed Time: {}".format(elapsed_time))

    id_list = [0] 
    plt.style.use('seaborn-white')


    for id_no in id_list:
        error_covariance_ellipse (torch.squeeze(X_train_KF).detach().cpu().numpy(),
                            torch.squeeze(y_train_KF).detach().cpu().numpy(), 
                            y_train_pred_mean.detach().cpu().numpy(), 
                            y_train_pred_logvar.detach().cpu().numpy(), 
                            batch_gaussian.detach().cpu().numpy(),
                            axs[plot_id],
                            id_no = id_no)


    # Using Dictionary to get rid of duplicate legend:
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), fontsize = 12, loc = 'lower left', bbox_to_anchor=( -0.05,-0.05) )

    # plt.savefig(f'./frames_goodwin/TIV_results/Results/{traj_folder}/prediction.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(f'./frames_goodwin/TIV_results/Results/{traj_folder}/prediction.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()

mus, sigmas = torch.stack(mu_preds), torch.stack(sigma_preds)
print(mus.shape)



# Compute the perfromance metrics:
ADE = average_displacement_error(mus.detach().cpu().numpy())
FDE = final_displacement_error(mus.detach().cpu().numpy())
print("ADE: {}".format(ADE))
print("FDE: {}".format(FDE))

# Relative Error between actual and estimated trajectory
error_ = relative_error( online_data[1],  online_data[2])
print("Relative Error: {}".format(error_))



# Compute the KL-divergence:
KL_div = []
H1, H2 = [], []
for pred in range(12):
    mu_1, sigma_1 = mus[1,0,pred,:2].detach().cpu().numpy(), np.diag(np.exp(sigmas[1,0,pred,:2].detach().cpu().numpy())) 
    mu_2, sigma_2 = mus[2,0,pred,:2].detach().cpu().numpy(), np.diag(np.exp(sigmas[2,0,pred,:2].detach().cpu().numpy()))
    Kl_div = KL_divergence(mu_1, mu_2, sigma_1, sigma_2)
    
    # Compute the entropy for two distributions:
    '''
    Do not change the order for np.exp and np.diag or else the off-diagonal terms will become 1 as np.exp(0) ==1
    '''
    h1, h2 = entropy(sigma_1), entropy(sigma_2) 

    
    H1.append(h1)
    H2.append(h2)  
    KL_div.append(Kl_div)

H1 = np.array(H1)
H2 = np.array(H2)  
 
KL_div = np.array(np.squeeze(KL_div))


KL_div_H = KL_div/H2
print("KL_div vs H:",KL_div)
print("KL_div avg:", np.mean(KL_div))

# print("mus :", mus[0,:,:,:2])
# print("Traj :", np.array(online_data)[0,:,:2])

