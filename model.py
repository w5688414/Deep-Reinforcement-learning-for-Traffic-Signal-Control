import torch.nn as nn
import torch
import torch.nn.functional as F


def orthogonal_init(layer, gain=1.0):
    if len(layer._parameters):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)

class Model(nn.Module):
    def __init__(self, model_type, args):
        super(Model, self).__init__()
        self.input_size = int(args.Number_States)
        self.Number_Actions = int(args.Number_Actions)
        self.hidden_size = int(args.hidden_size)
        self.model_type = model_type
        if args.output_type == 'descrete':
            self.output_layer_acti = nn.LogSoftmax(dim =-1)
        else:
            self.output_layer_acti = nn.Tanh()

        self.liner_model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
        )
        for i in range(args.hidden_layer):
            self.liner_model.add_module('layer_{}'.format(i), nn.Linear(self.hidden_size, self.hidden_size))
            self.liner_model.add_module('tanh_{}'.format(i), nn.Tanh())

        if self.model_type == 'actor':
            self.out_layer = nn.Linear(self.hidden_size, self.Number_Actions)
        else:
            self.out_layer = nn.Linear(self.hidden_size, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            for layer in self.liner_model:
                orthogonal_init(layer)
            if self.model_type == 'actor':
                
                orthogonal_init(self.out_layer, gain = 0.01)
            else:
                orthogonal_init(self.out_layer)


    def forward(self, x):
        x = self.liner_model(x)
        x = self.out_layer(x)
        if self.model_type == 'actor':
            x = self.output_layer_acti(x)
        return x
                
    def predict_one(self, state):
        state = torch.tensor(state).float()
        return torch.exp(self.forward(state)).detach().numpy()

    def predict_batch(self, states):
        states = torch.tensor(states).float()
        return self.forward(states).detach().numpy()

class ICMModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ICMModel, self).__init__()
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.state_dim = state_dim
    
    def forward(self, state, action, next_state):
        # print(f'state shape {state.shape}, action shape {action.shape}, next_state shape {next_state.shape}')
        # forward model
        predicted_next_state = self.forward_model(torch.cat([state, action], dim=1))
        # breakpoint()
        if predicted_next_state.shape != predicted_next_state.shape:
            breakpoint()
        forward_loss = F.mse_loss(predicted_next_state, next_state)
        
        # inverse model
        state_action = torch.cat([state, next_state], dim=1)
        predicted_action = self.inverse_model(state_action)
        inverse_loss = F.cross_entropy(predicted_action, action.argmax(dim=1))
        
        return predicted_next_state, forward_loss, inverse_loss
