
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
import torch.optim as optim
import random
import numpy as np

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


    def predict(self, input_tensor, target_len, device):
            
            '''
            : param input_tensor:      input data (seq_len, input_size); PyTorch tensor 
            : param target_len:        number of target values to predict 
            : return np_outputs:       np.array containing predicted values; prediction done recursively 
            '''
            input_tensor = input_tensor.permute(1, 0, 2) #batch_first=False
            # encode input_tensor
            encoder_output, encoder_hidden = self.encoder(input_tensor.to(device))

            # initialize tensor for predictions
            outputs = torch.zeros(target_len, input_tensor.shape[1], input_tensor.shape[2] +2, device=device) #target_len, B, 4

            # decode input_tensor
            decoder_input = input_tensor[-1, :, :]
    #         decoder_input_var = torch.ones_like(decoder_input)*min_var
    #         decoder_input = torch.cat([decoder_input, decoder_input_var], dim=1)
            decoder_hidden = encoder_hidden

            for t in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[t] = decoder_output
                decoder_input = decoder_output[:,:input_tensor.shape[2]]

            np_outputs = outputs #.detach() #.numpy()

            np_outputs = np_outputs.permute(1, 0, 2) #batch_first=False
            return np_outputs