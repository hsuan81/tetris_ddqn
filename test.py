# import Game.tetris_fun as game
import numpy as np
import cv2
import os
import gym
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

from collections import namedtuple
from DQN_train import *
from networks.models import *
from utils.replay_buffer import ReplayBuffer
from utils.plot import VisdomLinePlotter
os.environ["SDL_VIDEODRIVER"] = "dummy"
# plotter = VisdomLinePlotter(env_name='DQN training') 

def preprocess(image, shape=(84,84)):
    x_t = image
    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    x_t = x_t.copy().astype(np.float32) # image value must be float cause net needs the values to calculate gradients
    x_t = torch.from_numpy(x_t).unsqueeze(0)
    return x_t.unsqueeze(0)

def select_action(state, policy_net, n_actions):
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

# env = game.GameState()
# y, r, t = env.frame_step(np.array([0, 0, 1, 0, 0, 0]))
# print(y.shape)
# y1 = cv2.resize(y, (120, 120))
# print(y1.shape)
# cv2.imwrite("resize.png", y1)
# y2 = cv2.cvtColor(y1, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("grey.png", y2)
# ret, y3 = cv2.threshold(y2,1,255,cv2.THRESH_BINARY)
# cv2.imwrite("recolor.png", y3)
# cv2.waitKey(0)
# env.save_screen()

# game = gym.make('Tetris-v0')


# testing for the DQN


env = gym.make('CartPole-v0').unwrapped

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

# Necessary for unpacking the batch
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

# resize = T.Compose([T.Resize(40, interpolation=T.InterpolationMode.BICUBIC),
#                     T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)

def plot(episode, total_rewards, losses, epsilons=None):
        # clear_output(wait = True)
        # plt.figure.clear()
        plt.clf()
        
        
        plt.subplot(131)
        plt.title("episode: %s ,total_reward: %s" % (episode, total_rewards[episode - 1]))
        plt.plot(total_rewards)
        
        plt.subplot(132)
        plt.title("loss in each step")
        plt.plot(losses)
        
        plt.subplot(133)
        plt.title("epsilons in each step")
        # plt.plot(epsilons)
        
        plt.pause(0.01)
        

env.reset()
x = get_screen()
plt.ion()
plt.figure(figsize=(15, 10))
# print(x)

# cv2.imshow("Cartpole", get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy())

# plt.figure()
# plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
#            interpolation='none')
# plt.title('Example extracted screen')
# plt.show()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

steps_done = 0
num_actions = env.action_space.n
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
in_channels = 3
check_point = []
screen_shape = (screen_height, screen_width)
num_episodes = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(num_actions, in_channels, screen_shape).to(device)
target_net = DQN(num_actions, in_channels, screen_shape).to(device)
print("Built DQN")
    
# set weight and bias of target net as policy net
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(params=policy_net.parameters())

memory = ReplayBuffer(device, memory_size, screen_shape)

episode_durations = []
# scores = []
rewards = []
losses = []
epsilons = []
last_screen = []
timestep = 0
batch_update = 0

n_actions = env.action_space.n
for episode in range(num_episodes):
    survive_points = 0
    score = 0
    env.reset()

    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    
    while True:
        timestep += 1
        act = select_action(state, policy_net.to(device), n_actions)
        x_t1, r_1, terminal, info = env.step(int(act.item()))
        r_1 = torch.tensor([r_1], device=device)

        # print("ep %i time %i " %(episode, timestep))
        # print("state", env.heuristic_state())
        # print("reward", r_1)
        # print("action", act)
            

        # Tailor reward to guide the agent to survive
        # if not terminal:
        #     r_1 += 1
    
#             cv2.imwrite("frame"+str(timestep)+".png", x_t1)
        
        
        rewards.append(r_1)
        score += r_1
        # state = env.heuristic_state()
        # cleared_line = state[0]

        # plotter.plot('reward', 'train v3', 'Reward for Cartpole', 'timestep', timestep, r_1)
        # plotter.plot('score', 'train v3', 'Score for Cartpole', 'timestep', timestep, score)
        last_screen = current_screen
        current_screen = get_screen()
        if not terminal:
            next_state = current_screen - last_screen
            # print("screen shape", next_state.shape)
        else:
            next_state = None  # if it is terminal, the image is set to be whole black
            # print("next shape", next_state.shape)

        # terminal = torch.tensor(terminal)
        # print("shceck next", next_state.shape)
        memory.push(state, act, next_state, r_1, terminal)
        
        state = next_state
        

        # logger for inspection while training
        if timestep % 10 == 0:
            print("timestep", timestep)
        
        
        if memory.can_sample(BATCH_SIZE):
            batch_update += 1
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            
            state_batch = torch.cat(batch.state)
            
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            # print(batch.next_state[0].shape)
            # print("next state", batch.next_state[0].shape)
            next_state_batch = list(batch.next_state)
            done_mask = batch.done

            # print("next batch", next_state_batch[0].shape)

            # curr_qs = policy_net(state_batch.to(device))
            curr_qs = policy_net(state_batch.to(device)).gather(1, action_batch.to(device))  # retrieve the max Q-value according to the index specified in act-batch
            next_qs = get_next_qs(target_net, next_state_batch, done_mask, BATCH_SIZE, device)
            
            target_q_values = reward_batch + GAMMA * next_qs
            
            criterion = nn.SmoothL1Loss()
            loss = criterion(curr_qs, target_q_values.unsqueeze(1))  # loss is in Tensor type
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            
            plot(episode, rewards, losses)
            # plotter.plot('loss', 'train', 'Loss for Cartpole', 'Batch', batch_update, float(loss))
            
            # logger for inspection while training
            if batch_update % 10 == 0:
                print("batch", batch_update)

            # Update target net at even interval 
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        if terminal:
            episode_durations.append(timestep)
            # scores.append(score)

            # x_p = np.concatenate(list(x_t1), axis=1)
            # cv2.imwrite("train_ep" + str(episode) + "time" + str(timestep) + ".png", x_p)
            print("ep %i time %i" %(episode, timestep))
            print("score", score)
            
            
            # plotter.plot('score', 'train', 'Score for Cartpole', 'episode', episode, score)
            # plot_results(episode, scores, losses, num_episodes)
            # plot_durations(episode_durations)
            break
            
        
            
        # if episode in save_point:
        #     torch.save(policy_net, "%s/%s_%s.pth" % (saving_path, "DQN", episode))
        #     torch.save({
        #                 'episode': episode,
        #                 'model_state_dict': policy_net.state_dict(),
        #                 'target_state_dict': target_net.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'loss': loss,
        #                 'learning rate': lr,
        #                 }, "%s/%s_%s_train.pth" % (saving_path, "DQN", episode))   # save for later training

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()

