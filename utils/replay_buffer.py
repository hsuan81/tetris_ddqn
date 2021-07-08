import random
import torch
import numpy as np
from collections import deque 

class ReplayBuffer:
    
    def __init__(self, size, screen_shape=(84, 84), frame_stack=True):
        self.size = size
        self.screen_shape = screen_shape
        self.num_in_buffer = 0
        self.memory = deque(maxlen=self.size)
        
    def push(self, screen, action, reward, next_state, terminal):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        terminal = 0.0 if not terminal else 1.0
        screen = torch.from_numpy(screen).to(dtype=torch.float32, device=device)
        next_state = torch.from_numpy(next_state).to(dtype=torch.float32, device=device)
        action = action.to(device)
        reward = torch.tensor(reward).to(device)
        terminal = torch.tensor(terminal).to(device)
        self.memory.append((screen, action, reward, next_state, terminal))
        
        self.num_in_buffer = len(self.memory)
        
    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer
    
    def _encode_sample(self, idxes):
        # Return batch data for screens, actions, rewards, next screens and terminal info
        # one screen state corresponding to one action by default, needing to consider grouped screens and actions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples = [self.memory[i] for i in idxes]
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = zip(*samples)  # unzip the samples and return tuples
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = map(torch.stack, zip(*samples))  # convert tuple into tensor form
        
        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask
        
    
    def sample(self, batch_size):
#         assert self.can_sample(batch_size)
        inds = random.sample(range(self.num_in_buffer), batch_size)  
        
        return self._encode_sample(inds)