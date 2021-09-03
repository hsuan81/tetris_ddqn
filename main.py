from networks.models import *
from Game.tetris_env import *
from utils.replay_buffer import ReplayBuffer
from DQN_agentv2 import *

def main(reward, num_episodes, render):
    env = TetrisEnv()
    env = CropObservation(env, reduce_pixel=True, crop=True, board_width=8)
    if reward == 'baseline':
        env = HeuristicReward(env, ver=0)
        reward_ver = 0
    elif reward == 'fitness':
        env = HeuristicReward(env, ver=6)
        reward_ver = 6
    elif reward == 'action':
        env = HeuristicReward(env, ver=13)
        reward_ver = 13
    print("reward: {}".format(reward))
    
    
    in_channels = 1  # due to frame stack
    prioritised_exp = False
    render = render
    greedy = True
    random_agent = False
    n_actions = env.action_space.n
    screen_shape = env.shape
    device = torch.device("cuda" if torch.cuda.is_available() and use_GPU else "cpu")
    start_episode = 0


    policy_net = DQN(n_actions, in_channels=in_channels, screen_shape=screen_shape).to(device)
    target_net = DQN(n_actions, in_channels=in_channels, screen_shape=screen_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr)
    memory = ReplayBuffer(10000, screen_shape=screen_shape)
    resume = False
   

    # Training and testing
    check_point = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500,
            3000, 5000, 10000, 30000, 50000, 100000, 300000, 500000]
    record_point = [100, 150, 250, 350, 450, 600, 1000, 2000, 3500, num_episodes-50]
    
    
    
    train(env, 8, num_episodes, check_point, greedy=greedy, prioritised=prioritised_exp, 
        num_piece='all piece', render=render, train_ver=reward_ver, record_point=record_point, 
        start_episode=start_episode, resume_train=resume)

    trained = input("Enter the trained model path:")
    policy_net = torch.load(trained, map_location=torch.device(device))

    test(env, 1, policy_net, render=render)

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Use DQN to train Tetris by pixel')
    # positional
    parser.add_argument('--r', required=True, help='Use the reward specified')
    parser.add_argument('--eps', type=int, required=True,
                        help='Number of episode for training')

    # default
    parser.add_argument('--render', action="store_false", help='Render the game')

    args = parser.parse_args()
    main(reward=args.r, num_episodes=args.eps, render=args.render)