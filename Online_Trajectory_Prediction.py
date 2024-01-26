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

# Import classes
from kalman_module import *
from model import lstm_encoder, lstm_decoder, lstm_seq2seq
from Cooperative_tracking import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def error_covariance_ellipse(X_test, y_test, mus, sigmas, ground_cov,  id_no =100):
    

    num_fea = mus.shape[3]
    sigmas = np.exp(sigmas)
    mu_ens = np.mean(mus, axis=0)
    # sigma_ens = torch.sqrt((torch.sum(torch.square(mu_preds) + torch.square(sigma_preds),axis=0))/sigma_preds.shape[0] - torch.square(mu_ens))
#     var_ens = np.mean((sigmas + mus ** 2 ), axis = 0) - mu_ens**2
    var_aleatoric = np.mean(sigmas[:,:,:,:2], axis = 0)
    var_epistemic = np.mean(mus[:,:,:,:2]**2, axis = 0) - mu_ens[:,:,:2]**2
    var_ens = var_aleatoric  + var_epistemic
#     - mu_ens[:,:,:4]**2
#     var_epistemic = np.var(mus, axis = 0)
    var_state_unc = (np.mean((mus[:,:,:,2:4]), axis=0)) 
    ground_cov = ground_cov.transpose(1,0,2,3)
#     ground_cov = ground_cov.transpose(1,0,2,3)
#     print(ground_cov.shape)
#     var_state_unc_1 = np.mean(ground_cov[:,:,:,4:], axis = 0)

    
    #     + np.mean(sigmas[:,:,:,4:], axis = 0)
#     total_unc = var_ens[:,:,:2] + var_state_unc
#     print("Var_ens",var_ens)
#     print("var_tot",total_unc)
#     var_total = var_ens + var_state_unc
    
    # For a specific ID:
#     id_no = 20
#     var_ens = var_ens[:,:,:2]
#     var_state_unc = var_state_unc[:,:,:2]
#     var_state_unc_1 = var_state_unc_1[:,:,:2]


    ax.scatter(X_test[id_no,:,0],  X_test[id_no,:,1], color='r',marker='^',s =15, zorder=1)
    ax.scatter(y_test[id_no,:,0],  y_test[id_no,:,1], color='r', marker='^', s = 15, zorder=2)
    ax.plot(mu_ens[id_no,:,0], mu_ens[id_no,:,1], color='b', marker='d', alpha=0.85, ms = 5, label = 'NN state Estimate', zorder =3 )

    for pred in range(8):
#         ax.plot(train_traj[id,point,0], train_traj[id,point,1], lw= 4 -0.15*point, ms=12., marker='*', color="r", linestyle="dashed")

        cov = np.cov(ground_cov[:,id_no,pred,0],ground_cov[:,id_no,pred,1] ) 
        lambda_, v_ = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        # print(lambda_)
    #         print(lambda_tot)

        for j in range(2,3):
            ell3 = Ellipse(xy = (ground_cov[:,id_no,pred,0].mean(), ground_cov[:,id_no,pred,1].mean()),
                     width = (lambda_[0] ) * j* 2 ,
                     height = (lambda_[1] ) *j* 2,
                        angle = np.degrees(np.arctan2(v_[1, 0], v_[0, 0])),
                         color = 'black',  lw = 0.5) 
            ell3.set_facecolor('green')
            ell3.set_alpha(0.25)
            # ax.add_artist(ell3)
    ell3.set_label("KF state uncertainty $(2\sigma)$")
   
    
    state_cov = []
    pred_cov = []
    mu = []
    forward_pred = y_test.shape[1]
    for pred in range(forward_pred): 
        
        mean = np.squeeze(mu_ens[id_no, pred, :])
#         print(mean[0],mean[1])
        
        # Total Variance:
#         var_ens = np.squeeze(var_aleatoric[id_no, pred,:]).reshape(2,2) + np.diag(np.squeeze(var_epistemic[id_no, pred,:]))
        # Total Variance:
#         cov_epistemic = np.squeeze(var_epistemic[id_no, pred,:]))
        cov_pred = np.squeeze(np.squeeze(np.diag(var_ens[id_no,pred,:])))
#         print('cov_total', cov_pred)

        # Total Variance:
        cov_state = np.squeeze(var_state_unc[id_no, pred,:2])
        cov_state = np.diag(np.squeeze(cov_state))
#         print('cov_state',cov_state)
       
        lambda_tot, v_tot = np.linalg.eig(cov_pred)
        lambda_tot = np.sqrt(lambda_tot)
#         print(lambda_tot)
        
        lambda_ale, v_ale = np.linalg.eig(cov_state)
        lambda_ale = np.sqrt(lambda_ale)
#         print(lambda_ale)
        
        for j in range(1,2):
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

    print("mu", mu[:,:2])
    print("cov", pred_cov)
#     ax.scatter(X_test[id_no,:,0],  X_test[id_no,:,1], color='g',marker='o')
#     ax.plot(y_test[id_no,:,0],  y_test[id_no,:,1], color='r',alpha=0.5, marker='^' ,label = 'Ground Truth')
#     ax.plot(mu_ens[id_no,:,0], mu_ens[id_no,:,1], color='b', marker='d', alpha=0.85,  label = 'Predicted')
    ax.set_aspect('equal', adjustable='box')
    plt.rcParams.update({'font.size': 22})
    
    ax.set_xlim([-2,4])
    ax.set_xticks([0,2])
    ax.set_ylim([-2,7])
    csfont = {'fontname':'P052'}
#     hfont = {'fontname':'Nimbus Sans'}
    ell1.set_label("NN State Uncertainty $(2\sigma)$")
    ell2.set_label("NN Prediction Uncertainty $(\sigma$)")
    ax.set_ylabel('y (m)', **csfont,fontsize=16, fontweight ='normal')
    # ax.legend(loc = 'lower right')
    ax.set_xlabel('x (m)', **csfont,fontsize=16, fontweight ='normal')


#     plt.title('title',**csfont)
#     plt.xlabel('xlabel', **hfont)
#     plt.show()
#     elif a == ax[0]:
#         a.set_title('idx = %d' %idx)

    
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
        label.set_fontweight('normal')

    plt.grid("on", alpha = 0.25)
    plt.setp(ax.get_xticklabels(), fontsize=16)




# Run the cooperative Tracking of the pedestrian using cameras:
[traj_1, traj_1_transf, traj_2] = cooperative_estimation()


# Start time:
start_time = time.time()

data = [traj_1, traj_1_transf, traj_2]
train_traj = np.expand_dims(traj_1, axis =0)
train_traj = train_traj - train_traj[:,0,:]
# print(train_traj)
# X_train, y_train = np.split(train_traj, [8,12], axis = 1)


# Sampled Trajectory from KF posterior distribution: 
num_traj = 7 # Number of Sampled Trajectories
batch_traj_test, test_mu, test_cov =[], [], []
for id in range(train_traj.shape[0]):
    X_seq = train_traj[id,:,:]
    trajectories, mus, covs = get_N_trajectories(X_seq, num_traj)
    batch_traj_test.append(trajectories)
    test_mu.append(mus)
    test_cov.append(covs)

batch_traj, train_mu, train_cov = np.array(batch_traj_test), np.array(test_mu), np.array(test_cov)
print(batch_traj.shape)
# print(batch_traj.shape)


num_fea = 2
batch_traj = torch.tensor(batch_traj).to(device)
X_train_KF, y_train_KF = torch.split(batch_traj[:,:,:,:num_fea],[8,12],dim = 2)
batch_gaussian = np.concatenate([train_mu[:,:,:,:num_fea], train_cov[:,:,:,:num_fea]], axis = 3)
batch_gaussian = torch.tensor(batch_gaussian).float().to(device)
batch_gaussian_input, batch_gaussian_output = torch.split(batch_gaussian, [8,12], dim =2) 

# Load the MCD/DE saved NN model weights and biases as .pt file
PATH = './MCD_models/lstm_seq2seq_eth_zara01_zara02.pt'
num_fea = batch_gaussian_input.shape[3]
model = lstm_seq2seq(input_size=num_fea, hidden_size =128).to(device) 
# model.load_state_dict(torch.load(PATH))
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

idx = list(range(7))
Num_ens = 3
forward_pred = 12
look_back = 8
min_logvar, max_logvar = -4, 4
preds, sigmas = [],[]


for i in random.sample(idx,Num_ens):
    y_train_pred = model.predict(batch_gaussian_input[:,i,:,:], forward_pred, device)
    # print(y_train_pred.shape)
    y_train_pred_mu,   y_train_state_logvar, y_train_pred_logvar = y_train_pred[:,:,:int(num_fea/2)], (y_train_pred[:,:,int(num_fea/2):num_fea]), (y_train_pred[:,:,num_fea:])#target_len, b, 8
    y_train_state_var = torch.exp(y_train_state_logvar)
    y_train_pred_logvar = torch.clamp(y_train_pred_logvar, min=min_logvar, max=max_logvar)
    y_train_pred_mean = torch.cat((y_train_pred_mu, y_train_state_var),2)
    mse_train = ((y_train_pred_mean - batch_gaussian_output[:,i,:,:num_fea])**2).mean()
    #  print(f"Train MSE: {mse_train}")
    preds.append(y_train_pred_mean)
    sigmas.append(y_train_pred_logvar)

mu_preds, sigma_preds = torch.stack(preds), torch.stack(sigmas)  # Imp to convert a torch list to tensor

# Compute the time it takes to predict Trajectory:
end_time = time.time()
elapsed_time = start_time - end_time
print("Total Elapsed Time: {}".format(elapsed_time))

id_list = [0] 
plt.style.use('seaborn-white')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,3), sharex=False, sharey=False)

for id_no in id_list:
    error_covariance_ellipse (torch.squeeze(X_train_KF).detach().cpu().numpy() ,
                          torch.squeeze(y_train_KF).detach().cpu().numpy(), 
                          mu_preds.detach().cpu().numpy(), 
                          sigma_preds.detach().cpu().numpy(), 
                          batch_gaussian.detach().cpu().numpy(),
                          id_no = id_no)


# Using Dictionary to get rid of duplicate legend:
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), fontsize = 12, loc = 'lower left', bbox_to_anchor=( -0.05,-0.05) )

# plt.savefig('./frames_goodwin/Cam1_transf_occ_pred.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
# plt.savefig('/home/anshul/Downloads/ZED2_camera_ws/Cam_1_traj_pred.pdf', dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()

