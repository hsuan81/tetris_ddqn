import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
import pandas as pd
from datetime import datetime
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchsummary import summary

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

def select_action(observation, n_actions):
        """
        This function uses softmax action selection with continousily decreasing temperature. During training I have tried many variations of the temperature 
        and found that a decreasing version of this is the best way to takle between exploration and exploitation.
        Starting with higher temperature and decreasing the temperature so the probabilities will be sckewed to the highest probability.
        """
        global steps_done
        steps_done += 1
        with torch.no_grad():
            state = observation
            advantage = policy_net(state)
            # print("q val", advantage)
            soft = nn.Softmax(dim=-1)
            prob = soft(advantage).cpu().detach().numpy()[0]
            # prob = prob.cpu().detach().numpy()[0]
            action = np.random.choice(n_actions, p=prob)

        
        # if action == T.argmax(advantage).item():
        #     greedy.append(0)
        # else: 
        #     greedy.append(1)

        return torch.tensor([[action]], device=device)

# def select_action(state, n_actions):
#     global steps_done
#     sample = random.random()
#     eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#         math.exp(-1. * steps_done / EPS_DECAY)
#     # print("step", steps_done)
#     # print("eps threshold", eps_threshold)
#     steps_done += 1
#     if sample > eps_threshold:
#         # return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
#         with torch.no_grad():
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             print("policy net", policy_net(state))
#             # print("policy max", policy_net(state).max(1)[1].view(1, 1))
#             return policy_net(state).max(1)[1].view(1, 1)
#     else:
#         return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)



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
    plt.ylim(0, 1)
    plt.plot(losses)    
    
    plt.pause(0.01)

def get_torch_screen(np_screen):
    """ Convert numpty screen to Tensor. """
    screen = torch.from_numpy(np_screen).float().to(device)
    return screen.unsqueeze(0)

def store_data(path, data_title:str, data:list):
    data = {data_title: data}
    df = pd.DataFrame(data)
    df.to_csv(path)

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

    # Check q value approaching mechanism
    # print("sa v", state_action_values)
    # print("expect sa v", expected_state_action_values)
    # print("loss", type(loss))
    # print(loss, loss.item())

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()

def train(env, board_size, num_episodes, check_point, num_piece='all piece', render=False, train_ver=0, record_point=None, start_episode=0, resume_train=False):
    today = datetime.now()
    if not resume_train:
        saving_path = './model_saving/' + today.strftime('%m%d%H%M') + '_' + str(board_size)
    else:
        saving_path = './model_saving/' + today.strftime('%m%d%H%M') + '_resume' + str(board_size)
    model_dir = pathlib.Path(saving_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create new folder by date for data storage
    if not resume_train:
        path = "./results/" + today.strftime('%m%d%H%M') + '_' + str(board_size)
    else:
        path = "./results/" + today.strftime('%m%d%H%M') + '_resume' + str(board_size)
    new_dir = pathlib.Path(path)
    try: 
        new_dir.mkdir(parents=True, exist_ok=True)  # if parent path is not existing, create it 
    except FileExistsError as error: 
        print(error)

    para = ['board size: {}'.format(board_size), 
            'num of eps: {}'.format(num_episodes),
            'num of piece: {}'.format(num_piece), 
            'train ver: {}'.format(train_ver),
            'start episode: {}'.format(start_episode),
            'resume train: {}'.format(resume_train)]
    with open(path + '/parameter.txt', 'w') as f:
        f.writelines('\n'.join(para))

    cell_size = env.cell_size
    best_clear_lines = 0  # critetrion to store the best model 
    losses = []
    rewards = []
    each_reward = []
    cleared_lines = []
    episode_durations = []
    _temp_memory = []
    if record_point is not None:
        from gym.wrappers.monitoring import video_recorder
        if not resume_train:
            video_path = "./recording/" + today.strftime('%m%d%H%M') + '_' + str(board_size)
        else:
            video_path = "./recording/" + today.strftime('%m%d%H%M') + '_resume' + str(board_size)
        video_dir = pathlib.Path(video_path)
        try: 
            video_dir.mkdir(parents=True, exist_ok=True)  # if parent path is not existing, create it 
        except FileExistsError as error: 
            print(error)

    if not render:
        plotter = VisdomLinePlotter(today.strftime('%m%d%H%M'))
        
    for i_episode in range(1, num_episodes+1):
        # print("i_ep", i_episode)
        # Initialize the environment and state
        state = get_torch_screen(env.reset())
        # print("state shape", state.shape)
        total_reward = 0.0
        total_lines = 0
        
        record = False
        if record_point is not None:
            # from gym.wrappers.monitoring import video_recorder
            if i_episode in record_point:
                vid = video_recorder.VideoRecorder(env, path=video_path + "/vid_v%s_%s.mp4" %(train_ver, start_episode+i_episode))
                record = True
                re= 0
                print("record start:", start_episode+i_episode)
            elif record and re < 40:
                re += 1
            elif record and re == 40:
                vid.enabled = False
                vid.close()
                record = False
                print("record ends:", start_episode+i_episode)
        # print("episode", i_episode)
        for t in count():
            if state.dim() == 3:
                state = state.unsqueeze(0)
            # print("state", state[0][3])
            # Select and perform an action
            action = select_action(state, n_actions)
            # print("action", action)
            # action = env.sample()
            if render:
                env.render()
                time.sleep(0.02)
            if record:
                vid.capture_frame()
                re += 1

            next_state, reward, done = env.step(action.item())
            next_state = get_torch_screen(next_state)
            if next_state.dim() == 3:
                next_state = next_state.unsqueeze(0)
            # Obtain the heuristic state for this step
            h_state = env.heuristic_state()
            total_lines = env.total_lines()

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
            # memory.push(state, action, next_state, reward, done)
            _temp_memory.append((state, action, next_state, reward, done))


            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model()
            # Put loss into separate list 
            losses.append(loss)

            # if not render:
            #     plotter.plot("reward", "train", "reward per step", "step", t, reward.item())
            #     plotter.plot("loss", "train", "loss per step", "step", t, loss)

            if done or total_lines > 100:
                # @chi Memory more on success episodes
                for temp in _temp_memory:
                    memory.push(*temp)
                for _ in range(5 * total_lines):
                    for temp in _temp_memory:
                        memory.push(*temp)

                episode_durations.append(t + 1)
                rewards.append(total_reward)
                cleared_lines.append(total_lines)
                # print("train total", cl_lines)
                # plot_durations()
                plot(i_episode, rewards, each_reward, losses, cleared_lines, epsilons=None)
                if not render:
                    plotter.plot("rewards", "train", "rewards per ep", "episode", i_episode, total_reward)
                    plotter.plot("cleared lines", "train", "cleared lines per ep", "episode", i_episode, total_lines)
                break
            
            

            
        # Update the target network, copying all weights and biases in DQN 
        # modified from https://github.com/jmichaux/dqn-pytorch/blob/master/main.py
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if i_episode % 20 == 0:
            print('Total steps: {} \t Episode: {}/{} \t Total reward: {} \t lines: {}'.format(steps_done, start_episode+i_episode, t, total_reward, total_lines))

        if i_episode % 100 == 0:
            store_data(new_dir/'rewards_v{}.csv'.format(train_ver), "rewards by ep", rewards)
            store_data(new_dir/'loss_v{}.csv'.format(train_ver), "loss by step", losses)
            store_data(new_dir/'step_reward_v{}.csv'.format(train_ver), "reward by step", each_reward)
            store_data(new_dir/'lines_v{}.csv'.format(train_ver), "cleared lines by ep", cleared_lines)

                
        if i_episode in check_point:
            torch.save(policy_net, "%s/%s_%s_v%s.pth" % (saving_path, "DQN", start_episode+i_episode, train_ver))
            is_best = True if total_lines > best_clear_lines else False
            if is_best:
                best_clear_lines = total_lines
                torch.save({
                            'episode': start_episode+i_episode,
                            'model_state_dict': policy_net.state_dict(),
                            'target_state_dict': target_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'learning rate': lr,   # if it is Adam
                            'step': steps_done,
                            'best clear lines': best_clear_lines,
                            }, "%s/best_%s_%s_train_v%s.pth" % (saving_path, "DQN", start_episode+i_episode, train_ver))   # save for later training
            else:
                torch.save({
                            'episode': start_episode+i_episode,
                            'model_state_dict': policy_net.state_dict(),
                            'target_state_dict': target_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss,
                            'learning rate': lr,   # if it is Adam
                            'step': steps_done,
                            'best clear lines': best_clear_lines,
                            }, "%s/%s_%s_train_v%s.pth" % (saving_path, "DQN", start_episode+i_episode, train_ver))   # save for later training


    print('Complete')
    
    
    # Create new folder by date
    path = "./results/" + today.strftime('%m%d%H%M') + '_' + str(board_size)
    new_dir = pathlib.Path(path)
    try: 
        new_dir.mkdir(parents=True, exist_ok=True)  # if parent path is not existing, create it 
    except FileExistsError as error: 
        print(error)
    # Store results data
    data1 = {"rewards by ep": rewards}
    data2 = {"loss by step": losses}
    data3 = {"reward by step": each_reward}
    data4 = {"cleared lines by ep": cleared_lines}
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    df3 = pd.DataFrame(data3)
    df4 = pd.DataFrame(data4)
    p1 = 'rewards_v{}.csv'.format(train_ver)
    p2 = 'loss_v{}.csv'.format(train_ver)
    p3 = 'step_reward_v{}.csv'.format(train_ver)
    p4 = 'lines_v{}.csv'.format(train_ver)
    df1.to_csv(new_dir/p1)
    df2.to_csv(new_dir/p2)
    df3.to_csv(new_dir/p3)
    df4.to_csv(new_dir/p4)
    plt.savefig(path + '/DQN_ep{}_v{}.png'.format(num_episodes, train_ver), bbox_inches='tight')
    # Average cleared lines of last 100 games, if over some benchmark, save the model for testing
    last_100 = cleared_lines[-100:]
    avg_100 = sum(last_100) // len(last_100)
    if avg_100 > 50:
        try:
            file_name = saving_path + "{}/{}_{}_v{}.pth".format(saving_path, "DQN_avg100", avg_100, train_ver)
            torch.save(policy_net, file_name)
        except Exception:
            pass

    env.close()
    plt.ioff()
    # plt.show(block=True)

def test(env, n_episodes, policy_net, render=False, console=True):
    for episode in range(n_episodes):
        state = get_torch_screen(env.reset())
        if state.dim() == 3:
                state = state.unsqueeze(0)
        cl_lines = 0
        total_reward = 0.0
        for t in count():
            q_vals = policy_net(state)
            action = q_vals.max(1)[1].view(1,1)

            if render:
                env.render()
                time.sleep(0.02)

            if console:
                print("episode %s step %s" %(episode, t))
                print("state", state[0][-1])
                print("q value", q_vals)
                print("action", action.item())

            next_state, reward, done = env.step(action.item())
            next_state = get_torch_screen(next_state)
            if next_state.dim() == 3:
                next_state = next_state.unsqueeze(0)
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

    screen_crop = {
        6: (150, 110),
        8: (180, 150),
        10: (216, 200)
    }
    # Train for Tetris
    reward_ver = 13
    # reward_ver = 10
    board_size = 8
    piece_set = {0: 'all piece',
                 1: 'one piece',
                 2: 'two pieces'}
    piece_code = 0
    env = TetrisEnv()
    env = CropObservation(env, reduce_pixel=True, crop=True, board_width=board_size)  # 6x8: (150, 110) 10x12: (216, 200)
    env = HeuristicReward(env, ver=reward_ver)
    env = TetrisPreprocessing(env, frame_skip=0, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
    # env = FrameStack(env,4)

    plt.ion()

    BATCH_SIZE = 32
    GAMMA = 0.5  # 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200  # 1000000 originally
    TARGET_UPDATE = 10
    in_channels = 1  # due to frame stack
    lr = 0.001
    render = True
    


    # screen_shape = env.observation_space.shape[1:]
    screen_shape = env.shape  # the board size
    print("screen shape", screen_shape)

    # if gpu is to be used
    if torch.cuda.is_available():
        print("...GPU is using...")
        BATCH_SIZE = 64
        render = False
        print("batch size", BATCH_SIZE)
        print("render", render)
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
    
    # # DQN with all MLP structure
    # policy_net = DQN_MLP(n_actions, in_channels=in_channels, screen_shape=screen_shape).to(device)
    # target_net = DQN_MLP(n_actions, in_channels=in_channels, screen_shape=screen_shape).to(device)
    # target_net.load_state_dict(policy_net.state_dict())
    # target_net.eval()
  
    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(10000, screen_shape=screen_shape)
    resume = False
    start_episode = 0
    steps_done = 0

    # Training and testing
    plt.ion()
    plt.figure(figsize=(15, 10))
    num_episodes = 5000
    check_point = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500,
            3000, 5000, 10000, 30000, 50000, 100000, 300000, 500000]
    record_point = [100, 150, 250, 350, 450, 600, 1000, 2000, 3500, num_episodes-50]
    
    # Resume training setting
    # resume = True
    # saved_model = "model_saving/08110825_8/best_DQN_3000_train_v13.pth"
    # ckpt = torch.load(saved_model)
    # policy_net.load_state_dict(ckpt['model_state_dict'])
    # target_net.load_state_dict(ckpt['target_state_dict'])
    # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    # lr = ckpt['learning rate']
    # print("lr", lr)
    # steps_done = ckpt['step']
    # start_episode = 1000
    
    # Check the loading of learning rate
    for param_group in optimizer.param_groups:
        print("loaded lr", param_group['lr'])


    # summary(policy_net, (4, 10, 8), batch_size=-1)
    
    

    
    train(env, board_size, num_episodes, check_point, num_piece=piece_set[piece_code], render=render, train_ver=reward_ver, record_point=record_point, start_episode=start_episode, resume_train=resume)
    

    
    # reward_ver = 10
    # # restart env
    # train(env, board_size, num_episodes, check_point, render=True, train_ver=reward_ver,record_point=record_point)
    # torch.save(policy_net, "dqn_tetris_model")
    # policy_net = torch.load("model_saving/08071515_8/DQN_1000_v13.pth")

    # test(env, 5, policy_net, render=render)