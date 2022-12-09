import gym
import numpy as np
import random


#Actions: EAST (0), NORTH (1), SOUTH (2), WEST (3)

class Avoid(gym.Env):

    def __init__(self):
        self.reset()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(6)


    def at_east_border(self, x=None):
        if x != None:
            return x >= 4
        else:
            return self.x >= 4

    def at_north_border(self, y=None):
        if y!=None:
            return y >= 4
        else:
            return self.y >= 4

    def at_south_border(self, y=None):
        if y!=None:
            return y <= 0
        else:
            return self.y <= 0

    def at_west_border(self, x=None):
        if x != None:
            return x <= 0
        else:
            return self.x <= 0

    def is_collided(self):
        return (self.x == self.obst1_x and self.y == self.obst1_y) or (self.x == self.obst2_x and self.y == self.obst2_y)


    def move_obstacles(self):
        # obstacle 1
        obst1_action = random.randint(0,3)
        if obst1_action == 0 and self.at_east_border(self.obst1_x) == False:
            self.obst1_x += 1
        elif obst1_action == 1 and self.at_north_border(self.obst1_y) == False:
            self.obst1_y += 1
        elif obst1_action == 2 and self.at_south_border(self.obst1_y) == False:
            self.obst1_y -= 1
        elif obst1_action == 3 and self.at_west_border(self.obst1_x) == False:
            self.obst1_x -= 1
        # obstacle 2
        obst2_action = random.randint(0,3)
        if obst2_action == 0 and self.at_east_border(self.obst2_x) == False:
            self.obst2_x += 1
        elif obst2_action == 1 and self.at_north_border(self.obst2_y) == False:
            self.obst2_y += 1
        elif obst2_action == 2 and self.at_south_border(self.obst2_y) == False:
            self.obst2_y -= 1
        elif obst2_action == 3 and self.at_west_border(self.obst2_x) == False:
            self.obst2_x -= 1



    def step(self, action):
        if random.random() > self.slickness:
            # Apply acction
            if action == 0 and self.at_east_border() == False:
                self.x += 1
            elif action == 1 and self.at_north_border() == False:
                self.y += 1
            elif action == 2 and self.at_south_border() == False:
                self.y -= 1
            elif action == 3 and self.at_west_border() == False:
                self.x -= 1


        # Move obstacles
        self.move_obstacles()

        # Next Step
        self.steps+=1
        if self.is_collided() or self.steps >= self.MAX_STEPS:
            reward = 0
            done = True
        else:
            reward = 100
            done = False
        return self.__create_state(), reward, done, {}


    def __create_state(self):
        return np.array([self.x, self.y, self.obst1_x, self.obst1_y, self.obst2_x, self.obst2_y])


    def reset(self):
        self.steps = 0
        self.MAX_STEPS = 100
        self.slickness = 0.0 
        self.x = 0
        self.y = 0
        self.obst1_x = 4
        self.obst1_y = 4
        self.obst2_x = 4
        self.obst2_y = 4
        return self.__create_state()