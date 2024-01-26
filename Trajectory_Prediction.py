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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(lstm_encoder,self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = False)

    
    def forward(self, x):
        lstm_out, self.hidden = (self.lstm(x.view(x.shape[0], x.shape[1], self.input_size)))
        return lstm_out, self.hidden
    
    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                  torch.zeros(self.num_layers, batch_size, self.hidden_size))
        
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers =2):
        super(lstm_decoder,self).__init__()
        
        
        self.input_size =  input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,
                            num_layers = num_layers, batch_first = False)
        
        self.linear = nn.Linear(hidden_size,  input_size + 2)
#         self.linear = nn.Linear(hidden_size, input_size)
            
    
    def forward(self, x, encoder_hidden_states):
        
        # Shape of x is (N, features), we want (1,N, features)
        lstm_out, self.hidden = (self.lstm(x.unsqueeze(0), encoder_hidden_states))
        lstm_out = lstm_out
        output = self.linear(lstm_out.squeeze(0))
        
        return output, self.hidden




    
class lstm_seq2seq(nn.Module):
    def __init__(self, input_size, hidden_size):
        
        super(lstm_seq2seq, self).__init__()
        self.input_size =  input_size
        self.hidden_size = hidden_size
        
        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size).to(device)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size).to(device)
    
    
    def train_model(self, input_tensor, target_tensor, n_epochs, target_len, batch_size =64, beta = 0.5, 
                    training_prediction ='recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.001, dynamic_tf = False):
        
        '''
        input_tensor = input_data with shape (Batch, seq_length, input_size =4 )
        target_tensor = target_data with shape(Batch, target_length, output_size = 4)
        n_epochs = number of epochs
        training_prediction = type of prediction that the NN model has to perform either 'recursive' or 
                        student-teacher-forcing
        dynamic_tf =  dynamic tecaher forcing reduces the amount of teacher force ratio every epoch
        '''
        
        min_logvar, max_logvar = -4, 4

    
        # define optimizer
        optimizer = optim.Adam(self.parameters(), lr = learning_rate)
#         scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda = lambda epoch: 0.95)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                        factor=0.25, patience=5, threshold=0.001, threshold_mode='abs')
        
        # Initialize loss for each epoch
        losses = np.full(n_epochs, np.nan)
#            criterion = nn.MSELoss()
        
        # Number of batches:
        n_batch = int(input_tensor.shape[0]/batch_size)
        num_fea = input_tensor.shape[2]
        lrs = [] # Obtain the LR
        
        with trange(n_epochs) as tr:
            for it in tr:
                
                batch_loss = 0
                batch_loss_tf = 0
                batch_loss_no_tf = 0
                num_tf = 0
                num_no_tf = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.permute(1, 0, 2), batch_y.permute(1, 0, 2) # shape : seq, batch, input
                    
#                     print(batch_x.shape, batch_y.shape)
                    # Initialize output tensor:
                    outputs = torch.zeros(target_len, batch_y.shape[1], batch_y.shape[2] + 2).to(device)
                    
                    # Initialize the hidden state 
                    encoder_hidden = self.encoder.init_hidden(batch_y.shape[1])
                    
                    # zero the gradient:
                    optimizer.zero_grad()
                    
                    # Encoder outputs for the entire sequence of look-back:
                    encoder_output, encoder_hidden = self.encoder(batch_x)
                    
#                     print("encoder_output", encoder_output.shape)
#                     print("encoder hidden", encoder_hidden[0].shape)
#                     print("encoder cell state", encoder_hidden[1].shape)
                    
                    # Decoder outputs:
                    decoder_input = batch_x[-1,:,:] 
#                     decoder_input_var = torch.ones_like(decoder_input)*min_var
#                     decoder_input = torch.cat([decoder_input, decoder_input_var], dim=1)
#                     print(decoder_input.shape)
                    
                    decoder_hidden = encoder_hidden
#                     print(decoder_hidden[0].shape)
                    
                    if training_prediction == 'recursive':
                        # Predict recursively:
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output[:,:num_fea]
                            
                    if training_prediction == 'tecaher_forcing':
                        
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len):    
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = torch.squeeze(batch_y[t,:,:])
                        
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output[:,:num_fea]
                    
                    # compute the loss:
#                     outputs = outputs.to(device, dtype=torch.float64)
                    F, B, _ = outputs.shape
                    
                    #predictive UQ
                    outputs_mean, outputs_var = outputs[:,:,:int(num_fea/2)], outputs[:,:,num_fea:] #target_len, b, 8
#                     print( outputs_mean.shape, outputs_var.shape )
                    outputs_logvar = torch.clamp(outputs_var, min=min_logvar, max=max_logvar)
                    loss_NLL = 0.5*((outputs_mean - batch_y[:,:,:int(num_fea/2)])**2/torch.exp(outputs_logvar))+ 0.5*outputs_logvar
                    if beta > 0:
                        loss = loss_NLL * torch.exp(outputs_logvar).detach() ** beta
                    loss_NLL = torch.mean(loss)
                    
                    #State UQ
                    state_log_covar = (outputs[:,:,int(num_fea/2):num_fea])
                    state_covar = torch.exp(state_log_covar)
                    loss_MSE = torch.mean(0.5*(state_covar - batch_y[:,:,int(num_fea/2):num_fea])**2)
                    
                    loss = loss_NLL  + loss_MSE
                                                
#                     (torch.log(var) + ((y - mean).pow(2))/var).sum()
#                     loss = gaussian_nll(batch_y,outputs)
                    batch_loss += loss.item() # Compute the loss for entire batch 
                    
                    # Backpropagation:
                    loss.backward()
                    optimizer.step()
                    
                
                # LR scheduler:
                scheduler.step(loss)
#                 lrs.append(scheduler.get_last_lr())
                
                # Loss for epoch
                batch_loss /= n_batch
                losses[it]  = batch_loss
                
                # Dynamic teacher Forcing:
                if dynamic_tf and teacher_forcing_ratio >0:
                    teacher_forcing_ratio = teacher_forcing_ratio -0.02
                
                # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))
            
        return losses



def predict(model, input_tensor, target_len):
        
        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
        : param target_len:        number of target values to predict 
        : return np_outputs:       np.array containing predicted values; prediction done recursively 
        '''
        input_tensor = input_tensor.permute(1, 0, 2) #batch_first=False
        # encode input_tensor
        encoder_output, encoder_hidden = model.encoder(input_tensor.to(device))

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[1], input_tensor.shape[2] +2, device=device) #target_len, B, 4

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
#         decoder_input_var = torch.ones_like(decoder_input)*min_var
#         decoder_input = torch.cat([decoder_input, decoder_input_var], dim=1)
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            decoder_input = decoder_output[:,:input_tensor.shape[2]]

        np_outputs = outputs #.detach() #.numpy()

        np_outputs = np_outputs.permute(1, 0, 2) #batch_first=False
        return np_outputs


# Implement Kalman Filter for state estimation:
# X[k+1] = A * X[K] + B * u[K] + w

# Define the directories where your .pkl files are located
directory1 = './frames_goodwin/TIV_results/CAM_1/'
directory2 = './frames_goodwin/TIV_results/CAM_2/'

# Create an empty list to store the loaded data
data_list = []

# Loop through all files in first directory:


with open('./frames_goodwin/TIV_results/CAM_2/straight.pkl','rb') as f:
    data= pkl.load(f)

np.set_printoptions(precision =3)
# print(data)

# data = 1.5 * np.array(data)
# data = np.insert(arr = data, obj=[1],values = 0, axis =1)
# # print(data)

# # Do the transformation:
# R =  np.array([0.92768715, -0.04470923,  0.37067186,
# 0.02332297,  0.99780478,  0.06198113,
#  -0.37262929, -0.04885393,  0.92669343]).reshape(3,3)


# ### first transform the matrix to euler angles
# r =  Rotation.from_matrix(R)
# angles = r.as_euler("zyx",degrees=True)

# print("rpy", angles)

# for i in range(10):

#     sigma = np.random.normal(0, 0.5*np.abs(angles))
#     angles += 0 * sigma

# new_r = Rotation.from_euler("zyx",angles,degrees=True)
# new_rotation_matrix = new_r.as_matrix()



# t =  np.array([1.16323258, 0.0667586 ,0.04029998]).reshape(3,1)
# R,t = np.array(new_rotation_matrix),np.array(t)
# print("New_Rotation_Matrix", R)


# # print("Mat Mul:",np.matmul(R,data[1,:3].reshape(-1,1)))
# # print("Mat diff:",  (data[1,:3]))

# # If RON_12: convert 1 to coordinate frame of 2 : Inverse  transformation
# #  If RON_21: convert 2 to coordinate frame of 1: Rigid Transform

# coord_transf = np.zeros((data.shape[0],3))
# for i in range(data.shape[0]):
#     coord_transf[i,:] =  np.squeeze(np.matmul(R.T,(data[i,:3].reshape(-1,1)- t)))

# # print(coord_transf)

# data = np.concatenate((coord_transf[:,[0,2]],data[:,3:]), axis = 1)
# # print(data)
train_traj = np.expand_dims(data, axis =0)
train_traj = train_traj - train_traj[:,0,:]
# print(train_traj)
# X_train, y_train = np.split(train_traj, [8,12], axis = 1)

# print(X_train)
def get_F(dt):  
    F = np.array([[1., 0, dt, 0],
                  [0,  1., 0, dt],
                  [0,  0, 1., 0],
                  [0,  0, 0, 1]])
    return F

def get_Q(dt, var_wx, var_wy):
    Q = np.array([[0.25*dt**4*var_wx, 0, 0.5*dt**3*var_wx,0],
                  [0, 0.25*dt**4*var_wy, 0, 0.5*dt**3*var_wy],
                  [0.5*dt**3*var_wx, 0, dt**2*var_wx,0],
                  [0, 0.5*dt**3*var_wy,0, dt**2*var_wy]])
    return Q

# Define the process covariance matrix P:
sigma_x = 1
sigma_y = 1
sigma_u = 0.25
sigma_v = 0.25
P = np.diag([sigma_x**2,sigma_y**2, sigma_u**2, sigma_v**2]) 


def Kalman_filter(
                    X_prev, 
                    X_measured,
                    vx, vx_dot,
                    vy, vy_dot,
                    dt = 0.4,
                    var_wx = 0.5,
                    var_wy = 0.25,
                    P = P
                ):
    
    
#     noise_x = np.random.normal(mu, sigma, [X_prev.shape[0]])
#     X_prev_noise = X_train[:,:] + noise_x
#     x = X_prev_noise[0,:].T

    # Define the state transition matrix:
    F = get_F(dt)


    # Define Process noise:
    Q = get_Q(dt, var_wx, var_wy)
    
    # Define the update parameters:
    H = np.identity(X_prev.shape[0])
    
    # Define the measurement covariance:
    R = np.diag([vx**2, vy**2, vx_dot**2, vy_dot**2])
    
    # measurement 
    z = X_measured
    x = X_prev

    # Predict step:
    x = F @ x 
    P = F @ P @ F.T + Q

    # Update step of Kalman filter:
    # S = H*P*H.T + R ; R is the measurement covariance matrix
    S = H @ P @ H.T + R
    K = P @ H.T @ inv(S)
    y = z - H @ x

    x += K @ y
    P = P - K @ H @ P
    return x, P
        
    

def sample_distribution(mean, var, N=1):
    x_sample = np.random.multivariate_normal(mean, var, N)
    if N==1:
        x_sample = x_sample[0,:]
    return x_sample


def get_trajectory(
                    X_seq, # 8, 4
                    mu = 0.,
                    sigma = 0.05,
                    # add other inputs for Kalman filter
                    dt = 0.4,
                    var_wx = 0.5,
                    var_wy = 0.25
                  ):
    lookback = X_seq.shape[0]
    vx, vx_dot = 0.05*np.mean([X_seq[:,0]]), 0.05*np.mean([X_seq[:,2]])
    vy, vy_dot = 0.05*np.mean([X_seq[:,1]]), 0.05*np.mean([X_seq[:,3]]) 
    
    # Define the state variance, P:
    sigma_x = 0.5
    sigma_y = 0.5
    sigma_u = 0.25
    sigma_v = 0.25
    P = np.diag([sigma_x**2, sigma_y**2, sigma_u**2, sigma_v**2]) 
    
    trajectories, mus, covs = [], [], []
    for i in range(X_seq.shape[0]):
        X_measured = X_seq[i,:] 
        
        if i==0:
            X_prev = X_seq[0,:]
            # Add a sample noise to the first point
            noise_x = np.random.normal(mu, sigma, [X_prev.shape[0]])
            X_prev = X_prev + noise_x
#             X_prev = X_prev.T
        
        mu, cov = Kalman_filter(X_prev, 
                                X_measured, 
                                vx, 
                                vx_dot, 
                                vy, 
                                vy_dot,
                                #add other inputs
                                dt = 0.4,
                                var_wx = 0.5,
                                var_wy = 0.25,
                                P = P
                                )
        P = cov
        mus.append(mu)
        covs.append(cov.diagonal())
        x_new = sample_distribution(mu, cov, N=1)
        trajectories.append(x_new)
        X_prev = x_new
    
    mus = np.array(mus)
    covs = np.array(covs)
    trajectories = np.array(trajectories)
    return trajectories, mus, covs

def get_N_trajectories(
            X_seq,
            num_traj = 100,
            ):
    num_trajectories, batch_mu, batch_cov = [], [], []
    for i in range(num_traj):
        traj, mus, covs = get_trajectory(X_seq)
        num_trajectories.append(traj)
        batch_mu. append(mus)
        batch_cov.append(covs)
    num_trajectories = np.array(num_trajectories)
    batch_mu = np.array(batch_mu)
    batch_cov = np.array(batch_cov)
    
    return num_trajectories, batch_mu, batch_cov

num_traj = 7

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
    y_train_pred = predict(model, batch_gaussian_input[:,i,:,:], forward_pred)
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



def error_covariance_ellipse(X_test, y_test, mus, sigmas, ground_cov, id_no =100):
    

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



id_list = [0] 
print(plt.style.available)
# plt.style.use('seaborn-v0_8-white')
# plt.figure(figsize=(1,6))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4), sharex=False, sharey=True)


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

