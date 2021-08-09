import numpy as np
import os, subprocess, time, signal, copy
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import cv2
from collections import deque 

import logging
logger = logging.getLogger(__name__)

from random import randrange as rand
import pygame, sys

""" The board would look like this, and the last row is set as 1s.
[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]
"""

# The configuration
cell_size = 18  # original: 18
cols =      8  # standard game col is 10
rows =      10  # standard game row is 22
maxfps =    300


colors = [
(0,   0,   0  ),
(255, 85,  85),
(100, 200, 115),
(120, 108, 245),
(255, 140, 50 ),
(50,  120, 52 ),
(146, 202, 73 ),
(150, 161, 218 ),
(35,  35,  35) # Helper color for background grid
]

# Define the shapes of the single parts
tetris_shapes = [
    # [[1, 1, 1],
    #  [0, 1, 0]],

    # [[0, 2, 2],
    #  [2, 2, 0]],

    # [[3, 3, 0],
    #  [0, 3, 3]],

    # [[4, 0, 0],
    #  [4, 4, 4]],

    # [[0, 0, 5],
    #  [5, 5, 5]],

    [[6, 6, 6, 6]],

    [[7, 7],
     [7, 7]]
]

def rotate_clockwise(shape):
    return [
        [ shape[y][x] for y in range(len(shape)) ]
        for x in range(len(shape[0]) - 1, -1, -1)
    ]

def check_collision(board, shape, offset):
    """ Check whether the stone meets another. """
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[ cy + off_y][ cx + off_x]:
                    return True
            except IndexError:
                return True
    return False

def remove_row(board, row):
    del board[row]
    return [[0 for i in range(cols)]] + board

def join_matrixes(mat1, mat2, mat2_off):
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy+off_y-1][cx+off_x] += val
    return mat1

def new_board():
    board = [
        [ 0 for x in range(cols) ]
        for y in range(rows)
    ]
    board += [[ 1 for x in range(cols)]]
    return board

class TetrisApp(object):
    def __init__(self, show_next_stone=False):
        pygame.init()
        # pygame.key.set_repeat(250,25)
        self.cell_size = cell_size
        self.width = cell_size*(cols+6)
        self.height = cell_size*rows
        self.rlim = cell_size*cols
        self.bground_grid = [[ 8 if x%2==y%2 else 0 for x in range(cols)] for y in range(rows)]

        self.default_font =  pygame.font.Font(
            pygame.font.get_default_font(), 12)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION) # We do not need
                                                     # mouse movement
                                                     # events, so we
                                                     # block them.
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.show_next_stone = show_next_stone
        self.init_game()

    def new_stone(self):
        self.stone = self.next_stone[:]
        self.next_stone = tetris_shapes[rand(len(tetris_shapes))]
        self.stone_x = int(cols / 2 - len(self.stone[0])/2)
        self.stone_y = 0
        # self.stone_change = True

        if check_collision(self.board,
                           self.stone,
                           (self.stone_x, self.stone_y)):
            self.gameover = True  # game terminates if the new stone collide with other piece from start
            
            

    def init_game(self):
        self.board = new_board()
        self.board_width = len(self.board[0])
        self.board_height = len(self.board)-1
        self.new_stone()
        self.level = 1
        self.score = 0
        self.score_lev = 0   # This score is computed with level considered
        self.lines = 0
        self.cl_lines = 0
        self.fitness_val = 0
        self.filled = 0  # Grids filled within the 2-row size window
        self.stone_change = False
        self.gameover = False
        self.paused = False
        dont_burn_my_cpu = pygame.time.Clock()
        # pygame.time.set_timer(pygame.USEREVENT+1, 1000)  # let piece drop every 1000 milliseconds.

    def disp_msg(self, msg, topleft):
        x,y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(
                    line,
                    False,
                    (255,255,255),
                    (0,0,0)),
                (x,y))
            y+=14

    def center_msg(self, msg):
        for i, line in enumerate(msg.splitlines()):
            msg_image =  self.default_font.render(line, False,
                (255,255,255), (0,0,0))

            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2

            self.screen.blit(msg_image, (
              self.width // 2-msgim_center_x,
              self.height // 2-msgim_center_y+i*22))

    def draw_matrix(self, matrix, offset):
        # print("matrix", matrix)
        off_x, off_y  = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x+x) *
                              cell_size,
                            (off_y+y) *
                              cell_size,
                            cell_size,
                            cell_size),0)

    def add_cl_lines(self, n):
        """ Compute cleared line, score and level, and adjust the dropping spped as level goes up. """
        # linescores = [0, 40, 100, 300, 1200]  # Nintendo scoring system
        # linescores = [0, 10, 40, 90, 160]
        linescores = [0, 0, 0, 0, 0]
        self.cl_lines = n
        # print("cleared line", self.cl_lines)
        self.lines += n
        self.score += linescores[n]
        self.score_lev += linescores[n] * self.level
        if self.lines >= self.level*6:   # Level up
            self.level += 1
            # The dropping speed increases as level goes up
            # newdelay = 1000-50*(self.level-1)
            # newdelay = 100 if newdelay < 100 else newdelay
            # pygame.time.set_timer(pygame.USEREVENT+1, newdelay)

    def move(self, delta_x):
        if not self.gameover and not self.paused:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > cols - len(self.stone[0]):
                new_x = cols - len(self.stone[0])
                
            if not check_collision(self.board,
                                   self.stone,
                                   (new_x, self.stone_y)):
                self.stone_x = new_x
            # else:
            #     self.stone_change = True

    def quit(self):
        """ Exit the game. """
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()
    def check_after_drop(self):
        if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y+1)):
            self.stone_change = True
            self.board = join_matrixes(
                self.board,
                self.stone,
                (self.stone_x, self.stone_y))
            # print("after", self.board)
            self.new_stone()
            cleared_rows = 0
            while True:
                for i, row in enumerate(self.board[:-1]):
                    if 0 not in row:
                        self.board = remove_row(
                            self.board, i)
                        cleared_rows += 1
                        break
                else:
                    break
                # print("cleared rows", cleared_rows)
            self.add_cl_lines(cleared_rows)  # compute result
            return True
        return False

    def drop(self):
        """ The piece drop automatically if manual is False or drops forcely if True. """
        if not self.gameover and not self.paused:
            self.stone_y += 1
            # print("stone y", self.stone_y)
            # print("drop")

            # new_y = self.stone_y + 1
            # if not check_collision(self.board, self.stone, (self.stone_x, new_y)):
            #     self.stone_y = new_y

            if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y)):
                # print("before", self.board)
                self.stone_change = True
                self.board = join_matrixes(
                  self.board,
                  self.stone,
                  (self.stone_x, self.stone_y))
                # print("after", self.board)
                self.new_stone()
                cleared_rows = 0
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(
                              self.board, i)
                            cleared_rows += 1
                            break
                    else:
                        break
                # print("cleared rows", cleared_rows)
                self.add_cl_lines(cleared_rows)  # compute result
                return True
        return False

    def insta_drop(self):
        if not self.gameover and not self.paused:
            while(not self.drop()):
                pass

    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone
            # else:
            #     self.stone_change = True

                

    def toggle_pause(self):
        self.paused = not self.paused

    def reset(self):
        """ Reset the game. """
        self.init_game()

    def _check_height(self):
        row = self.stone_y + len(self.stone) - 1   # get the bottom row position of the stone
        col = self.stone_x
        col_len = len(self.stone[0])
        cols = []

        # Current stone was placed at the bottom
        if row + 1 == len(self.board) - 1:
            return row, cols

        for i in range(col_len):
            c = col + i
            if row < self.board_height and self.board[row][c] != 0:
                if self.board[row + 1][c] == 0:
                    cols.append(c)
                else:
                    cols = []
                    break
        next_row = row + 1
        # print("check height", (next_row, cols))
        return next_row, cols
        
    def row_window(self, min_h):
        bottom_r = self.board_height - min_h
        return self.board[bottom_r - 2 : bottom_r]

    def win_filled(self):
        _, _, min_h = self.total_height()
        # print("min h", min_h)
        win = self.row_window(min_h)
        # print("window", win)
        # print("board", self.board)
        filled = 0
        for r in win:
            for i in range(self.board_width):
                if r[i] != 0:
                    filled += 1

        # new_filled = filled - self.filled
        self.filled = filled
        return filled

    def landing_height(self):
        pass

    def falling_reward(self):
        value = 0
        # row_weight = [0] * (len(self.board) - 4) + [1, 1, 2, 2]  # simple version
        row_weight = [0.0, 0.0] + [0.5 ** x for x in range(self.board_height-2, 0, -1)]
        for cy, row in enumerate(self.stone):
            w = row_weight[cy + self.stone_y]
            c = sum(x > 0 for x in row)
            value += w * c
            # print("row value", value)
        return value


    def graph_value(self):
        value = 0
        w = 0
        board = copy.deepcopy(self.board[:-1])
        row_weight = [0] * (len(board) - 2) + [1, 2]
        # row_weight = [0, 0] + [i for i in range(1, len(self.board[0])+1)]
        # print("row w", row_weight)
        # check the current piece position
        if check_collision(self.board,
                               self.stone,
                               (self.stone_x, self.stone_y+1)):
            # print("add stone value")
            board = join_matrixes(board, self.stone, (self.stone_x, self.stone_y+1))
            # print("add to board", board)

        for i, row in enumerate(board):
            w = row_weight[i]
            c = sum(x > 0 for x in row)
            value += w * c

        
        
            # for cy, row in enumerate(self.stone):
            #     w = row_weight[self.stone_x + cy]
            #     c = sum(x > 0 for x in row)
            #     value += w * c
        return value

    def empty_cell(self):
        _, max_h, _ = self.total_height()
        # print("max h", max_h)
        # to_fill = max_h * self.board_width
        empty = 0
        top_row = self.board_height - max_h
        w = max_h
        # print("top row", top_row)
        for row in self.board[top_row:-1]:
            # empty += row.count(0)
            e = w * row.count(0)
            empty += e
            w -= 1        
        return empty

    def depth_wells(self):
        # sum of depth of wells
        wells = 0
        c = 0
        min_ys = []
        real_r, real_cs = self._check_height()

        for col in zip(*self.board):
            if c in real_cs:
                i = real_r
            else:
                i = 0
            while i < self.board_height and col[i] == 0:
                i += 1
            # print("pos", (c, i))
            min_ys.append(i)
            # wells covered (holes with consecutive empty cells cumulated vertically)
            if col[i:].count(0) > 1:
                head = None
                w = 0
                for x in range(i, self.board_height):
                    if col[x] == 0 and col[x-1] != 0:
                        head = x
                    if head is not None:
                        if col[x] != 0 and col[x-1] == 0: 
                            w = x - head
                        elif x == self.board_height-1 and col[x] == 0:
                            w = self.board_height - head
                        if w > 1:
                            wells += w 
                            w = 0
                            head = None

            c += 1
        
        # wells not covered
        ys = [x for x in enumerate(min_ys)]
        left = None
        for c, row in ys:
            if c == 0:
                if ys[c+1][1] <= row:
                    left = (c, 0)
                    # left = (c, self.board_height)
            elif ys[c-1][1] < row:
                left = (c, abs(ys[c-1][1] - row))
            if left is not None:
                if c == len(ys)-1:
                    if ys[c-1][1] <= row:
                        # right = (c, self.board_height)
                        right = (c, 0)
                        width = abs(left[0]-right[0]) + 1
                        # d = min(left[1], right[1]) * width
                        d = max(left[1], right[1]) * width
                        wells += d
                        left = None
                elif ys[c+1][1] < row:
                    right = (c, abs(ys[c+1][1] - row))
                    width = abs(left[0]-right[0]) + 1
                    # d = min(left[1], right[1]) * width
                    d = max(left[1], right[1]) * width
                    wells += d
                    left = None
        
        return wells



    def number_row_with_holes(self):
        row_with_holes = set()
        real_r, real_cs = self._check_height()
        r = real_r
        c = 0
        for col in zip(*self.board):
            if c in real_cs:
                r = real_r
            else:
                r = 0
            while r < self.board_height and col[r] == 0:
                r += 1
            for i in range(r, self.board_height):
                if col[i] == 0:
                    row_with_holes.add(i)
            c += 1

        return len(row_with_holes)

        

    def clear_lines(self):
        """ Return cleared lines in this step. """
        # print("call clear lines", self.cl_lines)
        return self.cl_lines
    
    def number_of_holes(self):
        '''Number of holes in the board (empty square with at least one block above it)'''
        holes = 0
        c = 0
        real_r, real_cs = self._check_height()
        for col in zip(*self.board):
            if c in real_cs:
                i = real_r
            else:
                i = 0
            while i < self.board_height and col[i] == 0:
                i += 1
            holes += len([x for x in col[i+1:] if x == 0])
            
            c += 1

        return holes

    def total_height(self):
        '''Sum and maximum height of the board'''
        sum_height = 0
        max_height = 0
        min_height = self.board_height

        c = 0
        real_r, real_cs = self._check_height()

        for col in zip(*self.board):
            if c in real_cs:
                i = real_r
            else:
                i = 0
            while i < self.board_height and col[i] == 0:
                i += 1
            height = self.board_height - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height
            c += 1

        return sum_height, max_height, min_height

    def bumpiness(self):
        '''Sum of the differences of heights between pair of columns'''
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        c = 0
        real_r, real_cs = self._check_height()

        for col in zip(*self.board):
            if c in real_cs:
                i = real_r
            else:
                i = 0
            while i < self.board_height and col[i] == 0:
                i += 1
            min_ys.append(i)

            c += 1
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += bumpiness

        return total_bumpiness, max_bumpiness

    

    def fitness_reward(self, ver=None):
        a, b, c, d = -0.51, 0.76, -0.36, -0.18
        lines = self.clear_lines()
        # print("fit reward", lines)
        holes = self.number_of_holes()
        agg_height, max_h, min_h = self.total_height()
        bumpiness, max_bump = self.bumpiness()
        space_filled = self.win_filled()
        num_row_holes = self.number_row_with_holes()
        wells = self.depth_wells()
        offset = ((1 + self.board_height) * self.board_height / 2) * self.board_width 

        if ver is None:
            # genetic algorithm in Stevens' paper
            new_fit = a * agg_height + b * lines + c * holes + d * bumpiness
        elif ver == 0:
            # cleared lines
            new_fit = 10 * lines
        elif ver == 1:
            # different coefficient for genetic algorithm
            new_fit = a * agg_height + b * lines + d * holes + c * bumpiness
        elif ver == 2:
            # Sum of height differences between adjacent columns
            d = -1
            new_fit = d * bumpiness
        elif ver == 3:
            # difference of space filled at the two rows above the lowest height and difference of max and min height
            new_fit = 0.5 * space_filled + a * (max_h - min_h)
        elif ver == 4:
            # difference of max and min height
            new_fit = -1 * (max_h - min_h)
        elif ver == 5:
            # punishment from termination
            new_fit = 0
        elif ver == 6:
            # number of holes
            new_fit = -1 * holes
        elif ver == 7:
            # aggregation of columns height
            new_fit = -1 * agg_height
        elif ver == 8:
            # rows with holes
            new_fit = -1 * num_row_holes
        elif ver == 9:
            new_fit = -1 * wells
        elif ver == 10:
            new_fit = -1 * max_h
        elif ver == 11:
            empty_c = self.empty_cell()
            offset = ((1 + self.board_height) * self.board_height / 2) * self.board_width 
            new_fit = -1 * empty_c + offset
            new_fit = new_fit / offset  # Normalisation

        elif ver == 12:
            normal = 24
            graph_val = self.graph_value()
            new_fit = graph_val / normal 
        
        elif ver == 13:
            falling_val = self.falling_reward()
            new_fit = falling_val


        # rew = new_fit - self.fitness_val
        rew = new_fit
        self.fitness_val = new_fit
        return rew  + lines

    def _combo_actions(self, move):
        if move == 0:
            self.rotate_stone()
            self.move(-1)
        elif move == 1:
            self.rotate_stone()
            self.move(1)
        elif move == 2:
            self.rotate_stone()
            self.insta_drop()
        elif move == 3:
            self.rotate_stone()
            self.drop()


    def _handle_actions(self, action):
        key_actions = {
            'NONE':         None, # self.drop(),
            'ESCAPE':       self.quit,  # not use
            'LEFT':         lambda:self.move(-1),
            'RIGHT':        lambda:self.move(+1),
            'DOWN':         lambda:self.drop(),
            'ROTATE':       self.rotate_stone,
            'ROTATE+LEFT':  lambda:self._combo_actions(0),
            'ROTATE+RIGHT': lambda:self._combo_actions(1),
            'ROTATE+INSD':  lambda:self._combo_actions(2),
            'ROTATE+DOWN':  lambda:self._combo_actions(3),
            'p':            self.toggle_pause,  # not use 
            'SPACE':        self.reset,  # not use
            'INSDROP':      self.insta_drop
        }

        action_type = ACTION_LOOKUP[action]
        # If the action is not no operation, take the action
        self.drop()
        # self.check_after_drop()
        if action > 0:
            key_actions[action_type]()
            # print("action done")
        # if not self.stone_change:
        #     print("no change")
        #     self.drop()
        #     self.check_after_drop()

        # if not self.stone_change:
        #     self.drop(False)  # piece goes down one step whenever one action(including no operation) was taken

        self.stone_change = False
        # for event in pygame.event.get():
        #         if event.type == pygame.QUIT:
        #             self.quit()
    
    def _draw_screen(self, train=True):
        self.screen.fill((0,0,0))
        if self.gameover and not train:  # if it's training, skip the game over page
            self.center_msg("""Game Over!\nYour score: %d
Press space to continue""" % self.score)
        else:
            if self.paused:
                self.center_msg("Paused")
            if self.gameover:
                self.center_msg("""Game Over!\nYour score: %d
Press space to continue""" % self.score)
            else:
                pygame.draw.line(self.screen,
                    (255,255,255),
                    (self.rlim+1, 0),
                    (self.rlim+1, self.height-1))
                if self.show_next_stone:
                    self.disp_msg("Next:", (
                        self.rlim+cell_size,
                        2))
                self.disp_msg("Score: %d\n\nLevel: %d\
\nLines: %d" % (self.score, self.level, self.lines),
                    (self.rlim+cell_size, cell_size*5))
                # self.draw_matrix(self.bground_grid, (100,100))
                self.draw_matrix(self.board, (0,0))
                self.draw_matrix(self.stone,
                    (self.stone_x, self.stone_y))
                if self.show_next_stone:
                    self.draw_matrix(self.next_stone,
                        (cols+1,2))

    def step(self, action=None):
        """ Return the current state, the reward and terminal status. """
#         self.screen.fill((0,0,0))
#         if self.gameover:
#             self.center_msg("""Game Over!\nYour score: %d
# Press space to continue""" % self.score)
#         else:
#             if self.paused:
#                 self.center_msg("Paused")
#             else:
#                 pygame.draw.line(self.screen,
#                     (255,255,255),
#                     (self.rlim+1, 0),
#                     (self.rlim+1, self.height-1))
#                 if self.show_next_stone:
#                     self.disp_msg("Next:", (
#                         self.rlim+cell_size,
#                         2))
#                 self.disp_msg("Score: %d\n\nLevel: %d\
# \nLines: %d" % (self.score, self.level, self.lines),
#                     (self.rlim+cell_size, cell_size*5))
#                 # self.draw_matrix(self.bground_grid, (100,100))
#                 self.draw_matrix(self.board, (0,0))
#                 self.draw_matrix(self.stone,
#                     (self.stone_x, self.stone_y))
#                 if self.show_next_stone:
#                     self.draw_matrix(self.next_stone,
#                         (cols+1,2))
        # if not self.gameover:
        if action is None:
            self._draw_screen()
        score_0 = self.score
        # print("score 0", score_0)
        # curr_line = self.lines
        # self._draw_screen()
        if self.cl_lines > 0:
            self.cl_lines = 0
        if action is not None:
            self._handle_actions(action)
            # print("take action")
        
        self._draw_screen()
        pygame.display.update()
        obs = self.get_screenRGB()
        # reward_line = self.lines - curr_line
        # print("score 1", self.score)
        reward = self.score - score_0
        done = self.is_gameover()

        return obs, reward, done

    def is_gameover(self):
        return self.gameover

    def get_cl_lines(self):
        """ Return current total number of cleared lines. """
        # print("total lines", self,lines)
        return self.lines

    def get_screenRGB(self):
        """ Return current screen in RGB format. """
        # screen = pygame.transform.rotate(screen, 90)
        screen = pygame.surfarray.array3d(
            pygame.display.get_surface()).astype(np.uint8)   
        screen = np.fliplr(np.rot90(screen, 1, (1, 0)))
        return  screen

    def render(self):
        return self.screen

    def run(self):
        key_actions = {
            'ESCAPE':   self.quit,
            'LEFT':     lambda:self.move(-1),
            'RIGHT':    lambda:self.move(+1),
            'DOWN':     lambda:self.drop(),
            'UP':       self.rotate_stone,
            'p':        self.toggle_pause,
            'SPACE':    self.reset,
            'RETURN':   self.insta_drop
        }

        self.gameover = False
        self.paused = False

        dont_burn_my_cpu = pygame.time.Clock()
        while 1:
            self.screen.fill((0,0,0))
            if self.gameover:
                self.center_msg("""Game Over!\nYour score: %d
Press space to continue""" % self.score)
            else:
                if self.paused:
                    self.center_msg("Paused")
                else:
                    pygame.draw.line(self.screen,
                        (255,255,255),
                        (self.rlim+1, 0),
                        (self.rlim+1, self.height-1))
                    # self.disp_msg("Next:", (
                    #     self.rlim+cell_size,
                    #     2))
                    # Display score and level
#                     self.disp_msg("Score: %d\n\nLevel: %d\
# \nLines: %d" % (self.score, self.level, self.lines),
#                         (self.rlim+cell_size, cell_size*5))
                    self.draw_matrix(self.bground_grid, (0,0))
                    self.draw_matrix(self.board, (0,0))
                    self.draw_matrix(self.stone,
                        (self.stone_x, self.stone_y))
                    # self.draw_matrix(self.next_stone,   # display the next piece
                    #     (cols+1,2))
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT+1:
                    self.drop()
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:
                    for key in key_actions:
                        if event.key == eval("pygame.K_"
                        +key):
                            key_actions[key]()

            dont_burn_my_cpu.tick(maxfps)


# The following code modified from 
# https://github.com/oscastellanos/gym-traffic/blob/master/gym_traffic/envs/TrEnv.py

class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}
    action_set = {
    0, # NOOP
    1, # LEFT
    2, # RIGHT
    3, # ROTATE
    4, # INSTANT DROP
    5, # DOWN
    6, # ROTATE+LEFT
    7, # ROTATE+RIGHT
    8, # ROTATE+INSD
    9  # ROTATE+DOWN
    }

    def __init__(self):
        self.game = TetrisApp()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255,
        shape=(self.game.height, self.game.width, 3), dtype=np.uint8)
        self.shape = (self.game.height, self.game.width)
        self.cell_size = self.game.cell_size
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def total_lines(self):
        return self.game.get_cl_lines()

    def cleared_lines(self):
        return self.game.clear_lines()

    def fitness_reward(self, ver=None):
        return self.game.fitness_reward(ver)

    def heuristic_state(self):
        cl_lines = self.game.clear_lines()
        # print("h state", cl_lines)
        holes = self.game.number_of_holes()
        height, _, _ = self.game.total_height()
        bumpiness, _ = self.game.bumpiness()
        window_filled = self.game.win_filled()
        num_row_hole = self.game.number_row_with_holes()
        wells = self.game.depth_wells()
        empty_cells = self.game.empty_cell()
        return (cl_lines, holes, height, bumpiness, window_filled, num_row_hole, wells, empty_cells)

    def step(self, action):
        """ 
        IMPORTANT: THE RETURN STATE IS IN PIXEL FORM, NOT NORMAL GYM STATE
        Return: state (the state after taking the action): numpy.ndarray,
                reward,
                done status (whether the game terminates)
        """
        ob, reward, done = self.game.step(action)

        return ob, reward, done

    def _get_obs(self):
        return self.game.get_screenRGB()


    def reset(self):
        """ Return the current state in pixel form. """
        self.game.reset()
        return self.game.step()[0]

    def _reset(self):
        self.game.reset()
        return self.game.step()

    # def _get_image(self):
    #     img = self._get_obs
    #     return img

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_obs()

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def close(self):
        self.game.quit()

class HeuristicReward(gym.RewardWrapper):
    def __init__(self, env, ver=None):
        super().__init__(env)
        self.ver = ver

    def step(self, action):
        ob, reward, done = self.env.step(action)
        fit_rew = self.env.fitness_reward(self.ver)
        return ob, self.reward(reward, fit_rew, done), done
    
    def reward(self, reward, fit_reward, done):
        done_r = 0 if not done else 0
        rew = reward + fit_reward + done_r
        return rew


# The code below was taken and slightly modified from 
# https://github.com/hardmaru/slimevolleygym/blob/master/slimevolleygym/slimevolley.py

class FrameStack(gym.Wrapper):
  def __init__(self, env, n_frames):
    """Stack n_frames last frames.
    (don't use lazy frames)
    modified from:
    stable_baselines.common.atari_wrappers
    :param env: (Gym Environment) the environment
    :param n_frames: (int) the number of frames to stack
    """
    gym.Wrapper.__init__(self, env)
    self.n_frames = n_frames
    self.frames = deque([], maxlen=n_frames)
    shp = env.observation_space.shape
    low = np.repeat(self.observation_space.low[np.newaxis, ...], self.n_frames, axis=0)
    high = np.repeat(self.observation_space.high[np.newaxis, ...], self.n_frames, axis=0)
    self.observation_space = spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)
    # self.observation_space = spaces.Box(low=0, high=255, shape=(n_frames, shp[0], shp[1], shp[2]),
    #                                     dtype=env.observation_space.dtype)

  def reset(self):
    obs = self.env.reset()
    for _ in range(self.n_frames):
        self.frames.append(obs)
    return self._get_obs()

  def step(self, action):
    obs, reward, done = self.env.step(action)
    self.frames.append(obs)
    return self._get_obs(), reward, done

  def _get_obs(self):
    """ 
    Return:
        numpy array with shape (width, height, 3 * n_frames)
    """
    assert len(self.frames) == self.n_frames
    # np.concatenate(list(self.frames), axis=1)  
    return np.stack(self.frames, axis=0)

def get_resized_ob(ob, board_width, cell_width, reduce_pixel=False):
    """ Crop the image to keep only the game board and reduce one cell to one pixel if required.
    Args:
    ob (2-dim list): observation to be modified.
    board_width (int): the board version of the Tetris game
    cell_width (int): the width for one grid in the game.
    reduce_pixel (bool): if True the observation will be reduced to one cell equivalent to one pixel, 
                        otherwise, the pixel number remains.
    """
    ob_edge = board_width*cell_width
    ob = [ r[:ob_edge] for r in ob]
    if reduce_pixel:
        reduced_board = []
        screen_width = len(ob[0])
        screen_height = len(ob)
        for r in range(0, screen_height, cell_width):
            row = ob[r]
            new_r = []
            for c in range(0, screen_width, cell_width):
                new_r.append(row[c])
            reduced_board.append(new_r)
        ob = reduced_board

    return np.array(ob)


# Discarded in this case
class CropObservation(gym.ObservationWrapper):
    r""" Crop and scale down the image observation to desirable size. 
    Args:
        env (Env): environment
        reduce_pixel (bool): if True one cell will be reduced to one pixel, otherwise, remain the same cell size.
        reduce_rate (int): the original width of one cell. 
        crop (bool): if True, then the score panel of the game is discarded, otherwise the panel remains. 
        board_width (int): the number of cell in one row. If the crop=False, this is None by default.
    """
    def __init__(self, env, reduce_pixel=False, crop=True, board_width=None):
        super(CropObservation, self).__init__(env)
        # if isinstance(shape, int):
        #     shape = (shape, shape)
        # assert all(x > 0 for x in shape), shape
        # self.shape = tuple(shape)
        
        self.shape = self.observation_space.shape
        self.reduce_pixel = reduce_pixel
        self.cell_width = env.cell_size
        # self.reduce_to = reduce_to
        self.board_width = board_width
        self.crop = crop

        if crop:
            width = self.cell_width * board_width
            height = self.shape[0]
            self.shape = (height, width, self.shape[2])
        if reduce_pixel:
            # width = (self.shape[1] // self.cell_width) * self.reduce_to
            # height = (self.shape[0]  // self.cell_width) * self.reduce_to
            width = (self.shape[1] // self.cell_width)
            height = (self.shape[0]  // self.cell_width)
            self.shape = (height, width, self.shape[2])
        # obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

        

    # def reset(self, **kwargs):
    #     observation = self.env.reset(**kwargs)
    #     return self.observation(observation)

    def step(self, action):
        observation, reward, done = self.env.step(action)
        return self.observation(observation), reward, done

    def reset(self):
        obs = self.env.reset()
        return self.observation(obs)

    def observation(self, observation):
        def reduce_to_one_pixel(cell_width, observation):
            ob = observation
            screen_width = len(ob[0])
            screen_height = len(ob)
            # print("reduce", screen_height, screen_width)
            ob = observation[:screen_height:cell_width, :screen_width:cell_width]  # reduce to one cell for one pixel
            # new_ob = []
            # for y, row in enumerate(ob):
            #     new_r = []
            #     for x in range(len(row)):
            #         if x % cell_width == 0:
            #             new_r.extend(row[x:x+reduce_to])
            #             print("new", new_r)
            #     new_ob.append(new_r)

            # return np.array(new_ob)
            return ob

        

        # print("shape", self.shape)
        # print("original shape", observation.shape)

        if self.reduce_pixel:
            observation = reduce_to_one_pixel(self.cell_width, observation)
        # print("reduced", observation.shape)
       
        if self.crop:
            observation = observation[:, :self.shape[1]]

        # print("croped", observation.shape)
        
        observation = np.array(observation)
        if observation.ndim == 2:
            observation = np.expand_dims(observation, -1)
        return observation

class TetrisPreprocessing(gym.Wrapper):
    def __init__(self, env, noop_max=0, frame_skip=4, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False):
        super().__init__(env)
        assert cv2 is not None, \
            "opencv-python package not installed! Try running pip install gym[atari] to get dependencies  for atari"
        assert frame_skip >= 0
        
        self.env = env
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs
        self.curr_ob = None
        self.cleared_lines_buffer = 0

        # buffer of most recent two observations for max pooling (not using in Tetris case)
        if grayscale_obs:
            self.obs_buffer = [np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                               np.empty(env.observation_space.shape[:2], dtype=np.uint8)]
        else:
            self.obs_buffer = [np.empty(env.observation_space.shape, dtype=np.uint8),
                               np.empty(env.observation_space.shape, dtype=np.uint8)]
        
        self.game_over = False

        _low, _high, _obs_dtype = (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        _shape = self.observation_space.shape[:2] + (1 if grayscale_obs else 3,)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = spaces.Box(low=_low, high=_high, shape=_shape, dtype=_obs_dtype)
        self.shape = self.observation_space.shape

    def step(self, action):
        R = 0.0
        self.cleared_lines_buffer = 0

        for t in range(self.frame_skip if self.frame_skip > 0 else 1):
            self.curr_ob, reward, done = self.env.step(action)
            # print("t reward", reward)
            # print(self.env.heuristic_state())
            R += reward
            self.game_over = done
            self.cleared_lines_buffer += self.env.cleared_lines()

            if done:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    pass
                else:
                    pass
            elif t == self.frame_skip - 1:
                # if self.grayscale_obs:
                #     # self.ale.getScreenGrayscale(self.obs_buffer[0])
                #     pass
                # else:
                    # # self.ale.getScreenRGB2(self.obs_buffer[0])
                    pass
        return self._get_obs(), R, done

    def heuristic_state(self):
        cl_lines = self.cleared_lines_buffer
        h_state = self.env.heuristic_state()
        new_h_state = (cl_lines,) + h_state[1:]
        # print("h state", cl_lines)
        # holes = self.game.number_of_holes()
        # height, _, _ = self.game.total_height()
        # bumpiness, _ = self.game.bumpiness()
        # window_filled = self.game.win_filled()
        # num_row_hole = self.game.number_row_with_holes()
        # wells = self.game.depth_wells()
        
        return new_h_state  # (cl_lines, holes, height, bumpiness, window_filled, num_row_hole, wells)

    def reset(self, **kwargs):
        # NoopReset
        self.curr_ob = self.env.reset()
        noops = np.randint(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()

        if self.grayscale_obs:
            ob = self._get_obs()
        else:
            ob = self.env.reset()
        # self.obs_buffer[1].fill(0)
        return ob

    def _get_obs(self):
        # if self.frame_skip > 1:  # more efficient in-place pooling
        #     np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = self.curr_ob
        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            ret, obs = cv2.threshold(obs,1,255,cv2.THRESH_BINARY)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs


# Modified from https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
class DiscardTetrisPreprocessing(gym.Wrapper):
    r"""Atari 2600 preprocessings. 
    This class follows the guidelines in 
    Machado et al. (2018), "Revisiting the Arcade Learning Environment: 
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:
    * NoopReset: obtain initial state by taking random number of no-ops on reset. 
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional
    Args:
        env (Env): environment
        noop_max (int): max number of no-ops
        frame_skip (int): the frequency at which the agent experiences the game. 
        screen_size (int): resize Atari frame
        terminal_on_life_loss (bool): if True, then step() returns done=True whenever a
            life is lost. 
        grayscale_obs (bool): if True, then gray scale observation is returned, otherwise, RGB observation
            is returned.
        grayscale_newaxis (bool): if True and grayscale_obs=True, then a channel axis is added to
            grayscale observations to make them 3-dimensional.
        scale_obs (bool): if True, then observation normalized in range [0,1] is returned. It also limits memory
            optimization benefits of FrameStack Wrapper.
        crop_obs (bool): if True, crop the image to keep necessary observation, otherwise, original observation 
            is returned
    """

    def __init__(self, env, noop_max=0, frame_skip=4, screen_size=84, grayscale_obs=True,
                 grayscale_newaxis=False, scale_obs=False, crop_obs=True):
        super().__init__(env)
        assert cv2 is not None, \
            "opencv-python package not installed! Try running pip install gym[atari] to get dependencies  for atari"
        assert frame_skip >= 0
        if resize:
            assert screen_size > 0
        
        self.env = env
        self.noop_max = noop_max
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.grayscale_obs = grayscale_obs
        self.grayscale_newaxis = grayscale_newaxis
        self.scale_obs = scale_obs
        self.crop_obs = crop_obs
        self.curr_ob = None
        self.cleared_lines_buffer = 0

        # buffer of most recent two observations for max pooling (not using in Tetris case)
        if grayscale_obs:
            self.obs_buffer = [np.empty(env.observation_space.shape[:2], dtype=np.uint8),
                               np.empty(env.observation_space.shape[:2], dtype=np.uint8)]
        else:
            self.obs_buffer = [np.empty(env.observation_space.shape, dtype=np.uint8),
                               np.empty(env.observation_space.shape, dtype=np.uint8)]
        
        self.game_over = False

        _low, _high, _obs_dtype = (0, 255, np.uint8) if not scale_obs else (0, 1, np.float32)
        _shape = (screen_size, screen_size, 1 if grayscale_obs else 3)
        if grayscale_obs and not grayscale_newaxis:
            _shape = _shape[:-1]  # Remove channel axis
        self.observation_space = spaces.Box(low=_low, high=_high, shape=_shape, dtype=_obs_dtype)

    def step(self, action):
        R = 0.0
        self.cleared_lines_buffer = 0

        for t in range(self.frame_skip if self.frame_skip > 0 else 1):
            self.curr_ob, reward, done = self.env.step(action)
            # print("t reward", reward)
            # print(self.env.heuristic_state())
            R += reward
            self.game_over = done
            self.cleared_lines_buffer += self.env.cleared_lines()

            if done:
                break
            if t == self.frame_skip - 2:
                if self.grayscale_obs:
                    pass
                else:
                    pass
            elif t == self.frame_skip - 1:
                # if self.grayscale_obs:
                #     # self.ale.getScreenGrayscale(self.obs_buffer[0])
                #     pass
                # else:
                    # # self.ale.getScreenRGB2(self.obs_buffer[0])
                    pass
        return self._get_obs(), R, done

    def heuristic_state(self):
        cl_lines = self.cleared_lines_buffer
        h_state = self.env.heuristic_state()
        new_h_state = (cl_lines,) + h_state[1:]
        # print("h state", cl_lines)
        holes = self.game.number_of_holes()
        height, _, _ = self.game.total_height()
        bumpiness, _ = self.game.bumpiness()
        window_filled = self.game.win_filled()
        num_row_hole = self.game.number_row_with_holes()
        wells = self.game.depth_wells()
        
        return new_h_state  # (cl_lines, holes, height, bumpiness, window_filled, num_row_hole, wells)

    def reset(self, **kwargs):
        # NoopReset
        self.curr_ob = self.env.reset()
        noops = np.randint(1, self.noop_max + 1) if self.noop_max > 0 else 0
        for _ in range(noops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()

        # if self.grayscale_obs:
        #     ob = self._get_obs()
        # else:
        #     ob = self.env.reset()
        # self.obs_buffer[1].fill(0)
        return self._get_obs()

    def _get_obs(self):
        # if self.frame_skip > 1:  # more efficient in-place pooling
        #     np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])
        obs = cv2.resize(self.curr_ob, (self.screen_size, self.screen_size), interpolation=cv2.INTER_AREA)
        
        if self.scale_obs:
            obs = np.asarray(obs, dtype=np.float32) / 255.0
        else:
            obs = np.asarray(obs, dtype=np.uint8)

        if self.grayscale_obs:
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            ret, obs = cv2.threshold(obs,1,255,cv2.THRESH_BINARY)

        if self.grayscale_obs and self.grayscale_newaxis:
            obs = np.expand_dims(obs, axis=-1)  # Add a channel axis
        return obs


ACTION_LOOKUP = {
    0 : 'NONE',
    1 : 'LEFT',
    2 : 'RIGHT',
    # 3 : 'DOWN',
    3 : 'ROTATE', # Used on defense to slide tackle the ball
    # 5 : 'INSDROP',  # Used only by goalie to catch the ball
    # 6 : 'ROTATE+LEFT',
    # 7 : 'ROTATE+RIGHT',
    # 8 : 'ROTATE+INSD',
    # 9 : 'ROTATE+DOWN'
}

def print_table(ob):
    for o in ob:
        print(o, end='\n')

# For testing
if __name__ == '__main__':
    import os
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    FRAMESTACK = True
    PREPROCESS = True
    from gym.wrappers.monitoring import video_recorder
    # enable logging
    gym.logger.set_level(gym.logger.DEBUG)
    screen_crop = {
        (8, 6): (150, 110),
        (10, 8): (10, 8),#(180, 150),
        (12, 10): (216, 200)
    }

    
    if FRAMESTACK:
        game = TetrisEnv()
        game = CropObservation(game, reduce_pixel=True, reduce_to=3, crop=True, board_width=cols)
        game = HeuristicReward(game, ver=13)
        game = TetrisPreprocessing(game, frame_skip=0, grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)
        # game = FrameStack(game,4)
        vid = video_recorder.VideoRecorder(game,path="./recording/vid_test.mp4")
        # game = gym.wrappers.Monitor(game, "./recording/vid_test.mp4", video_callable=lambda episode_id: True,force=True)
        
        print("Test begins")
        x_t1 = game.reset()
        vid.capture_frame()
        print("test1",game.observation_space.shape)
        print("test1-1",x_t1.shape)
        # ob = np.concatenate(ob, axis=1)
        # print(ob.shape)
        if not PREPROCESS:
            x_t1 = cv2.cvtColor(x_t1, cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (288, -1, 1))
            print("no preprocess")
        # print(x_t1.shape)
        print(x_t1)
        # print(get_resized_ob(x_t1, cols, cell_size, reduce_pixel=True))
        # print_table(get_resized_ob(x_t1, cols, cell_size, reduce_pixel=True))
        # x_t1 = np.concatenate(x_t1, axis=1) 
        cv2.imwrite("framestack" + ".png", x_t1)
        for i in range(50):
            # action = game.action_space.sample()
            # action = 4
            # vid.capture_frame()
            game.render()
            action = int(input())
            print("%i action %i" % (i,action))
            x_t2, reward1, done1 = game.step(action)
            # print("count", (np.unique(x_t2, return_counts=True)))
            print("reward", reward1)
            # print(x_t2.shape)
            if not PREPROCESS:
                x_t2 = cv2.cvtColor(x_t2, cv2.COLOR_BGR2GRAY)
            # ret, x_t2 = cv2.threshold(x_t2,1,255,cv2.THRESH_BINARY)
            # x_t2 = np.reshape(x_t2, (288, -1, 1))
            # print_table(get_resized_ob(x_t2, cols, cell_size, reduce_pixel=True))
            print(x_t2)
            # x_t2 = np.concatenate(list(x_t2), axis=1)
            cv2.imwrite("stackframe" + str(i) + ".png", x_t2)
            h_state = game.heuristic_state()
            print(i, h_state)
            if reward1 > 0:
                print("score", reward1)
            if done1:
                x_t2 = game.reset()
            
            
            # print(game.observation_space.shape)
            
    else:
        game = TetrisEnv()
        # game = CropObservation(game, (216, 216))
        print("Test begins")
        ob = game.reset()
        print(game.observation_space)
        print(ob.shape)
        x_t1 = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (ob.shape[0], -1, 1))
        cv2.imwrite("frame" + ".png", x_t1)

        for i in range(50):
            game.render()
            # action = game.action_space.sample()
            # action = 4
            action = int(input())
            print("%i action %i" % (i,action))
            ob1, reward1, done1 = game.step(action)
            h_state = game.heuristic_state()
            print(i, h_state, game.total_lines())
            print("fit reward", game.fitness_reward)
            x_t2 = cv2.cvtColor(ob1, cv2.COLOR_BGR2GRAY)
            # ret, x_t2 = cv2.threshold(x_t2,1,255,cv2.THRESH_BINARY)
            # x_t2 = np.reshape(x_t2, (288, -1, 1))
            cv2.imwrite("frame" + str(i) + ".png", x_t2)
            print(ob1.shape)
            print(game.observation_space.shape)
            if done1:
                ob1 = game.reset()
            

    



