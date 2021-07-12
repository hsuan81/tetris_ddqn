import gym, math, glob, random
import numpy as np

import cv2
from timeit import default_timer as timer
import torch
from networks.models import *
from Game.tetris_env import *
from utils.replay_buffer import ReplayBuffer
from utils.plot import VisdomLinePlotter

# Solution for error: no available video device
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.001
SEED = 20

global plotter
plotter = VisdomLinePlotter(env_name='DQN training') 
random.seed(SEED)


def get_action(state, policy_net):
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
            
            state = torch.from_numpy(state)
            if state.ndim < 4:
                state = state.unsqueeze(0)
            action = policy_net(state.to(device, dtype=torch.float32))
            action = action.max(1)[1]
            print("net action")

            return action
    else:
        action = torch.tensor([random.randint(0, 5)])
        print("random action")

        return action


def get_next_qs(target_net, next_obs_batch, done_mask, BATCH_SIZE):
    """
    Return the Q-value of the next state.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terminal_ind = torch.nonzero(done_mask)
    values = target_net(next_obs_batch.to(device)).max(dim=1)[0].detach() 
    values[terminal_ind] = 0.0
    
    return values

def train(env, num_actions, in_channels, memory_size=100000, screen_shape=(84, 84), target_update=10, 
          BATCH_SIZE=128, GAMMA=0.999, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=0.001, lr=0.001, num_episodes=1000,
         save_point=[500, 1000, 3000, 5000, 10000, 50000, 100000]):
    env = env
    saving_path = './model_saving'
    save_point = save_point
    
    # if GPU is available, use it otherwise use CPU
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(num_actions, in_channels, screen_shape).to(device)
    target_net = DQN(num_actions, in_channels, screen_shape).to(device)
    print("Built DQN")
    
    # set weight and bias of target net as policy net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    
    memory = ReplayBuffer(memory_size, screen_shape)
    
    episode_durations = []
    scores = []
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
            act = get_action(x_t0, policy_net)
            x_t1, r_1, terminal = env.step(int(act))
            print("ep %i time %i " %(episode, timestep))
            print("state", env.heuristic_state())
            print("reward", r_1)
            print("action", act)
            x_p = np.concatenate(list(x_t1), axis=1)
            cv2.imwrite("train_ep" + str(episode) + "time" + str(timestep) + ".png", x_p)

            # Tailor reward to guide the agent to survive
            if not terminal:
                r_1 += 1
        
#             cv2.imwrite("frame"+str(timestep)+".png", x_t1)
            
            # Add extra reward if the agent survive
            score += r_1

            
            
            memory.push(x_t0, act, r_1, x_t1, terminal)
            
            # print("score", score)
            
            x_t0 = x_t1

            # logger for inspection while training
            if timestep % 10 == 0:
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

                # plotter.plot('loss', 'train', 'Loss', 'Batch', batch_update, float(loss))
                
                # logger for inspection while training
                if batch_update % 100 == 0:
                    print("batch", batch_update)

                # Update target net at even interval 
                if timestep % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                
            if terminal:
                episode_durations.append(timestep)
                scores.append(score)

                print("score", score)
                
                
                # plotter.plot('score', 'train', 'Score', 'episode', episode, score)
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
TARGET_UPDATE = 100
FRAMESKIP = 4
lr = 0.0001
memory_size = 100000
num_episodes = 10
check_point = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
            3000, 5000, 10000, 50000, 100000, 300000, 500000]

env = TetrisEnv()
env = HeuristicReward(env)
env = TetrisPreprocessing(env, screen_size=84)
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
