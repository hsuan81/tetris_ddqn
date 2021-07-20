import gym, math, glob, random
import numpy as np

import cv2
from timeit import default_timer as timer
import torch
from collections import namedtuple
from networks.models import *
from Game.tetris_env import *
from utils.replay_buffer import ReplayBuffer
from utils.plot import VisdomLinePlotter

# Solution for error: no available video device
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.001
SEED = 20

global plotter
plotter = VisdomLinePlotter(env_name='DQN training') 
random.seed(SEED)

steps_done = 0
def select_action(device, state, policy_net, n_actions=6):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

def get_action(state, policy_net, num_actions=6):
    # Return a number indicating the pos of 1 in the array for a action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps_done = 0
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state)
            if state.ndim < 4:
                state = state.unsqueeze(0)
            action = policy_net(state.to(device, dtype=torch.float32))
            action = action.max(1)[1]
            # print("net action")

            return action
    else:
        action = torch.tensor([random.randint(0, num_actions-1)])
        # print("random action")

        return action


def get_next_qs(target_net, next_obs_batch, done_mask, BATCH_SIZE, device):
    """
    Return the Q-value of the next state.
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # terminal_ind = torch.nonzero(done_mask)
    # values = target_net(next_obs_batch.to(device)).max(dim=1)[0].detach() 
    # values[terminal_ind] = 0.0
    
    
    non_final_indx = [i for i in range(len(done_mask)) if done_mask[i] is not True]
    print("done m", non_final_indx[:8])
    non_final_next_states = torch.cat([next_obs_batch[i] for i in non_final_indx])
    # print("next", non_final_next_states[0].shape)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_indx] = target_net(non_final_next_states).max(1)[0].detach()
    
    return next_state_values

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # print(batch.next_state[0].shape)
    # print(type(batch.next_state))                                 
    state_batch = torch.cat(batch.state)
    
    # print("state shape", state_batch.shape)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print("loss", loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

def train(env, num_actions, in_channels, memory_size=100000, screen_shape=(84, 84), target_update=10, 
          BATCH_SIZE=128, GAMMA=0.999, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=0.001, lr=0.001, num_episodes=1000,
         save_point=[500, 1000, 3000, 5000, 10000, 50000, 100000]):
    env = env
    saving_path = './model_saving'
    save_point = save_point
    
    # if GPU is available, use it otherwise use CPU
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(device, num_actions, in_channels, screen_shape).to(device)
    target_net = DQN(device, num_actions, in_channels, screen_shape).to(device)
    print("Built DQN")
    
    # set weight and bias of target net as policy net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.RMSprop(params=policy_net.parameters())
    
    memory = ReplayBuffer(device, memory_size, screen_shape)
    
    episode_durations = []
    scores = []
    rewards = []
    losses = []
    last_screen = []
    timestep = 0
    batch_update = 0
    
    for episode in range(num_episodes):
        survive_points = 0
        score = 0
        x_t0 = env.reset()
        
        while True:
            timestep += 1
            act = select_action(device, x_t0, policy_net, num_actions)
            x_t1, r_1, terminal = env.step(int(act))
            # print("ep %i time %i " %(episode, timestep))
            # print("state", env.heuristic_state())
            # print("reward", r_1)
            # print("action", act)
                

            # Tailor reward to guide the agent to survive
            # if not terminal:
            #     r_1 += 1
        
#             cv2.imwrite("frame"+str(timestep)+".png", x_t1)
            
            # Add extra reward if the agent survive
            score += r_1
            rewards.append(r_1)
            state = env.heuristic_state()
            cleared_line = state[0]

            plotter.plot('reward', 'train v3', 'Reward for 10X10', 'timestep', timestep, r_1)
            plotter.plot('cleared line', 'train v3', 'Lines for 10X10', 'timestep', timestep, cleared_line)
            
            memory.push(x_t0, act, x_t1, r_1, terminal)
            
            
            x_t0 = x_t1

            # logger for inspection while training
            if timestep % 100 == 0:
                print("timestep", timestep)
            
            
            if memory.can_sample(BATCH_SIZE):
                batch_update += 1
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = memory.sample(BATCH_SIZE)
                
                curr_qs = policy_net(obs_batch.to(device))
                curr_qs = policy_net(obs_batch.to(device)).gather(1, act_batch.to(device))  # retrieve the max Q-value according to the index specified in act-batch
                next_qs = get_next_qs(target_net, next_obs_batch, done_mask, BATCH_SIZE)
                
                target_q_values = rew_batch + GAMMA * next_qs
                
                criterion = nn.MSELoss()
                loss = criterion(curr_qs, target_q_values.unsqueeze(1))  # loss is in Tensor type
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                plotter.plot('loss', 'train v3', 'Loss for 10X10', 'Batch', batch_update, float(loss))
                
                # logger for inspection while training
                if batch_update % 100 == 0:
                    print("batch", batch_update)

                # Update target net at even interval 
            if episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())
            
            if terminal:
                episode_durations.append(timestep)
                scores.append(score)

                x_p = np.concatenate(list(x_t1), axis=1)
                cv2.imwrite("train_ep" + str(episode) + "time" + str(timestep) + ".png", x_p)
                print("ep %i time %i" %(episode, timestep))
                print("score", score)
                
                
                plotter.plot('score', 'train v3', 'Score for 10x10', 'episode', episode, score)
                # plot_results(episode, scores, losses, num_episodes)
                # plot_durations(episode_durations)
                break
                
            
                
            if episode in save_point:
                torch.save(policy_net, "%s/%s_%s.pth" % (saving_path, "DQN", episode))
                torch.save({
                            'episode': episode,
                            'model_state_dict': policy_net.state_dict(),
                            'target_state_dict': target_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'learning rate': lr,
                            }, "%s/%s_%s_train.pth" % (saving_path, "DQN", episode))   # save for later training


num_actions = 6
in_channels = 4  # due to frame stack
screen_shape = (84, 84)

BATCH_SIZE = 64
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.001
TARGET_UPDATE = 200
FRAMESKIP = 4
lr = 0.0001
memory_size = 100000
num_episodes = 100000
check_point = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
            3000, 5000, 10000, 50000, 100000, 300000, 500000]

env = TetrisEnv()
env = HeuristicReward(env, ver=3)
env = TetrisPreprocessing(env, screen_size=84, frame_skip=2)
env = FrameStack(env,4)


# env.reset()
# o, r, d = env.step(1)
# print(o, r, d)
# y, r, t = env.frame_step(np.array([0, 0, 1, 0, 0, 0]))
# print(y)
# x = env.get_pixelscreen()
# env.save_screen()


train(env, num_actions, in_channels, memory_size, screen_shape, 
    target_update = TARGET_UPDATE, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, 
    EPS_START=EPS_START, EPS_END=EPS_END, EPS_DECAY=EPS_DECAY, lr=lr, 
    num_episodes=num_episodes, 
    save_point=check_point)
