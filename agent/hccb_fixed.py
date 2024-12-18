import hcc

class fixed_CB(hcc.HCCAgent):
    def __init__(self, name, obs_type, obs_shape, action_shape, hidden_dim, lr, gamma, batch_size, tau, update_every_steps, device, use_wandb, nstep, sequence_length, epsilon):
        super().__init__(name, obs_type, obs_shape, action_shape, hidden_dim, lr, gamma, batch_size, tau, update_every_steps, device, use_wandb, nstep, sequence_length, epsilon)
    
    def fix_cb(self):
        for param in self.cerebellum_net.parameters():
            param.requires_grad = False
        