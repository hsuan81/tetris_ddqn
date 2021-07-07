import Game.tetris_fun as game
import numpy as np
import cv2
import os
import gym
import gym_tetris
os.environ["SDL_VIDEODRIVER"] = "dummy"

def preprocess(image, shape=(84,84)):
    x_t = image
    x_t = cv2.cvtColor(cv2.resize(x_t, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    x_t = x_t.copy().astype(np.float32) # image value must be float cause net needs the values to calculate gradients
    x_t = torch.from_numpy(x_t).unsqueeze(0)
    return x_t.unsqueeze(0)

env = game.GameState()
y, r, t = env.frame_step(np.array([0, 0, 1, 0, 0, 0]))
print(y.shape)
y1 = cv2.resize(y, (120, 120))
print(y1.shape)
cv2.imwrite("resize.png", y1)
y2 = cv2.cvtColor(y1, cv2.COLOR_BGR2GRAY)
cv2.imwrite("grey.png", y2)
ret, y3 = cv2.threshold(y2,1,255,cv2.THRESH_BINARY)
cv2.imwrite("recolor.png", y3)
cv2.waitKey(0)
env.save_screen()

game = gym.make('Tetris-v0')