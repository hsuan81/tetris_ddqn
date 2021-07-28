import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from networks.models import *
from Game.tetris_env import *
from utils.replay_buffer import ReplayBuffer
from utils.plot import VisdomLinePlotter

# set up os for docker container
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# Helping function for Cartpole env
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

# def get_screen():
#     # Returned screen requested by gym is 400x600x3, but is sometimes larger
#     # such as 800x1200x3. Transpose it into torch order (CHW).
#     screen = env.render(mode='rgb_array').transpose((2, 0, 1))
#     # Cart is in the lower half, so strip off the top and bottom of the screen
#     _, screen_height, screen_width = screen.shape
#     screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
#     view_width = int(screen_width * 0.6)
#     cart_location = get_cart_location(screen_width)
#     if cart_location < view_width // 2:
#         slice_range = slice(view_width)
#     elif cart_location > (screen_width - view_width // 2):
#         slice_range = slice(-view_width, None)
#     else:
#         slice_range = slice(cart_location - view_width // 2,
#                             cart_location + view_width // 2)
#     # Strip off the edges, so that we have a square image centered on a cart
#     screen = screen[:, :, slice_range]
#     # Convert to float, rescale, convert to torch tensor
#     # (this doesn't require a copy)
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
#     screen = torch.from_numpy(screen)
#     # Resize, and add a batch dimension (BCHW)
#     return resize(screen).unsqueeze(0)

def select_action(state, n_actions):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)



# def plot_durations():
#     plt.figure(2)
#     plt.clf()
#     durations_t = torch.tensor(episode_durations, dtype=torch.float)
#     plt.title('Training...')
#     plt.xlabel('Episode')
#     plt.ylabel('Duration')
#     plt.plot(durations_t.numpy())
#     # Take 100 episode averages and plot them too
#     if len(durations_t) >= 100:
#         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
#         means = torch.cat((torch.zeros(99), means))
#         plt.plot(means.numpy())

#     plt.pause(0.001)  # pause a bit so that plots are updated
#     if is_ipython:
#         display.clear_output(wait=True)
#         display.display(plt.gcf())

def plot(episode, total_rewards, rewards, losses, cleared_lines, epsilons=None):
    # plt.figure(figsize=(10, 10))
    plt.clf()
    
    
    plt.subplot(221)
    plt.title("episode: %s ,total_reward: %s" % (episode, total_rewards[episode - 1]))
    plt.xlabel("episode")
    plt.plot(total_rewards)

    plt.subplot(222)
    plt.title("cleared lines")
    plt.xlabel("episode")
    plt.plot(cleared_lines)
    
    plt.subplot(223)
    plt.title("reward: %s" % rewards[-1])
    plt.xlabel("step")
    plt.plot(rewards)

    plt.subplot(224)
    plt.title("loss")
    plt.xlabel("step")
    plt.ylim(0, 20)
    plt.plot(losses)    
    
    plt.pause(0.01)

def get_torch_screen(np_screen):
    screen = torch.from_numpy(np_screen).float().to(device)
    return screen.unsqueeze(0)


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

def train(env, num_episodes, check_point, render=False, train_ver=0, record_point=None):
    saving_path = './model_saving'
    losses = []
    rewards = []
    each_reward = []
    cleared_lines = []
    episode_durations = []
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state = get_torch_screen(env.reset())
        # print("state shape", state.shape)
        total_reward = 0.0
        cl_lines = 0
        
        record = False
        if record_point is not None:
            from gym.wrappers.monitoring import video_recorder
            if i_episode in record_point:
                vid = video_recorder.VideoRecorder(env, path="./recording/vid_v%s_%s.mp4" %(train_ver, i_episode))
                record = True
            elif i_episode in [x+50 for x in record_point]:
                vid.enabled = False
                vid.close()
                record = False

        for t in count():
            # Select and perform an action
            action = select_action(state, n_actions)
            # action = env.sample()
            if render:
                env.render()
                time.sleep(0.02)
            if record:
                vid.capture_frame()

            next_state, reward, done = env.step(action.item())
            next_state = get_torch_screen(next_state)
            # Obtain the heuristic state for this step
            h_state = env.heuristic_state()
            cl_lines += h_state[0]

            total_reward += reward
            each_reward.append(reward)

            # turn reward to tensor form
            reward = torch.tensor([reward], device=device)

            # Observe new state
            # last_screen = current_screen
            # current_screen = get_screen()
            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward, done)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model()
            # Put loss into separate list 
            losses.append(loss)

            if done:
                episode_durations.append(t + 1)
                rewards.append(total_reward)
                cleared_lines.append(cl_lines)
                # print("train total", cl_lines)
                # plot_durations()
                plot(i_episode, rewards, each_reward, losses, cleared_lines, epsilons=None)
                break
            
            

            
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 20 == 0:
            print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t lines: {}'.format(steps_done, i_episode, t, total_reward, cl_lines))

        

                
        if i_episode in check_point:
                torch.save(policy_net, "%s/%s_%s_v%s.pth" % (saving_path, "DQN", i_episode, train_ver))
                torch.save({
                            'episode': i_episode,
                            'model_state_dict': policy_net.state_dict(),
                            'target_state_dict': target_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            # 'learning rate': lr,   # if it is Adam
                            }, "%s/%s_%s_train_v%s.pth" % (saving_path, "DQN", i_episode, train_ver))   # save for later training

    print('Complete')
    # Average cleared lines of last 100 games, if over some benchmark, save the model for testing
    last_100 = cleared_lines[-100:]
    avg_100 = sum(last_100) // len(last_100)
    if avg_100 > 0:
        torch.save(policy_net, "%s/%s_%s.pth" % (saving_path, "DQN_avg100", avg_100))
    # 
    plt.savefig('%s/DQN_ep%s_v%s.png' % ('results', num_episodes, train_ver), bbox_inches='tight') 
    env.close()
    plt.ioff()
    # plt.show(block=True)

def test(env, n_episodes, policy_net, render=False):
    for episode in range(n_episodes):
        state = get_torch_screen(env.reset())
        cl_lines = 0
        total_reward = 0.0
        for t in count():
            action = policy_net(state).max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            next_state, reward, done = env.step(action.item())
            next_state = get_torch_screen(next_state)
            h_state = env.heuristic_state()
            cl_lines += h_state[0]

            total_reward += reward

            if done:
                next_state = None

            state = next_state

            if done:
                print("Finished Episode {}/{} with reward {} and cleared lines {}".format(episode, t, total_reward, cl_lines))
                break

    print("Complete")
    env.close()
    return

if __name__ == '__main__':
    # Test for CartPole env
    # env = gym.make('CartPole-v0').unwrapped

    # Train for Tetris
    reward_ver = 7
    env = TetrisEnv()
    env = CropObservation(env, (216, 200))
    env = HeuristicReward(env, ver=reward_ver)
    env = TetrisPreprocessing(env, screen_size=84, frame_skip=2)
    env = FrameStack(env,4)

    plt.ion()

    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200  # 1000000 originally
    TARGET_UPDATE = 200
    in_channels = 4  # due to frame stack
    lr = 0.0001
    check_point = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 
            3000, 5000, 10000, 50000, 100000, 300000, 500000]


    screen_shape = env.observation_space.shape[1:]
    # print("screen shape", screen_shape)

    # if gpu is to be used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Code and hyperparameter for CartPole
    # env.reset()

    # BATCH_SIZE = 128
    # GAMMA = 0.999
    # EPS_START = 0.9
    # EPS_END = 0.05
    # EPS_DECAY = 200
    # TARGET_UPDATE = 10

    # # Get screen size so that we can initialize layers correctly based on shape
    # # returned from AI gym. Typical dimensions at this point are close to 3x40x90
    # # which is the result of a clamped and down-scaled render buffer in get_screen()
    # init_screen = get_screen()
    # _, _, screen_height, screen_width = init_screen.shape
    # screen_shape = (screen_height, screen_width)

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    print("num action", n_actions)


    policy_net = DQN(n_actions, in_channels=in_channels, screen_shape=screen_shape).to(device)
    target_net = DQN(n_actions, in_channels=in_channels, screen_shape=screen_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayBuffer(10000, screen_shape=screen_shape)

    steps_done = 0

    # Training and testing
    plt.ion()
    plt.figure(figsize=(15, 10))
    num_episodes = 700
    record_point = [300, 500, num_episodes-50]
    train(env, num_episodes, check_point, render=True, train_ver=reward_ver)
    # torch.save(policy_net, "dqn_tetris_model")
    # policy_net = torch.load("model_saving/DQN_600.pth")

    # test(env, 20, policy_net, render=True)