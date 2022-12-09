import gym
import numpy as np
import random


#Actions: DROP (0), EAST (1), NORTH (2), PICK_UP (3), SOUTH (4), WEST (5)

class Transporter(gym.Env):

    def __init__(self):
        self.reset()
        self.action_space = gym.spaces.Discrete(6)
        self.observation_space = gym.spaces.Discrete(8)


    def is_passenger_on_board(self):
        return self.passenger == True

    def at_east_border(self):
        return self.x >= 4

    def at_north_border(self):
        return self.y >= 4

    def at_south_border(self):
        return self.y <= 0

    def at_west_border(self):
        return self.x <= 0

    def at_pick_up_location(self):
        return self.x == self.passenger_loc_x and self.y == self.passenger_loc_y

    def at_drop_location(self):
        return self.x == self.passenger_dest_x and self.y == self.passenger_dest_y

    def get_normal_reward(self):
        return 21 + (max(self.passenger_dest_x-self.x, self.x-self.passenger_dest_x-self.x) + max(self.passenger_dest_y-self.y, self.y - self.passenger_dest_y)) * (-1)

    def get_out_of_fuel_reward(self):
        return -100

    def __reset_passenger(self):
        red_x = 0
        red_y = 4
        yellow_x = 0
        yellow_y = 0
        green_x = 4
        green_y = 4
        blue_x = 3
        blue_y = 0
        locs = [(red_x, red_y),(yellow_x, yellow_y),(green_x, green_y),(blue_x, blue_y)]

        n_loc = random.choice(locs)
        n_dest = random.choice(locs)

        self.passenger_loc_x = n_loc[0]
        self.passenger_loc_y = n_loc[1]
        self.passenger_dest_x = n_dest[0]
        self.passenger_dest_y = n_dest[1]


    def step(self, action):
            if action == 0:

                if self.at_pick_up_location() and self.is_passenger_on_board() == False:
                    reward = 21 * (-1)
                else:
                    reward = self.get_normal_reward()
                    
            elif action == 1 and self.at_east_border() == False:
                self.x += 1
                reward = self.get_normal_reward()

            elif action == 2 and self.at_north_border() == False:
                self.y += 1
                reward = self.get_normal_reward()

            elif action == 3:
                if self.at_drop_location() and self.is_passenger_on_board():
                    self.jobs_done+=1
                    reward = 0
                    self.__reset_passenger()
                else:
                    reward = self.get_normal_reward()

            elif action == 4 and self.at_south_border() == False:
                self.y -= 1
                reward = self.get_normal_reward()

            elif action == 5 and self.at_west_border() == False:
                self.x -= 1
                reward = self.get_normal_reward()

            else:
                reward = self.get_normal_reward()
            info = {}
            self.fuel-=1
            if self.jobs_done == self.MAX_JOBS:
                done = True
            else:
                done = False

            if self.fuel == 0:
                reward = self.get_out_of_fuel_reward()
                done = True

            return self.__create_state(), reward, done, info

    def __create_state(self):
        passenger = 1 if self.passenger == True else 0
        return np.array([self.x, self.y, self.jobs_done, passenger, self.passenger_loc_x, self.passenger_loc_y, self.passenger_dest_x, self.passenger_dest_y, self.fuel])


    def reset(self):
            # {"fuel": 20, "jobs_done": 0, "passenger": 0, "passenger_dest_x": 0, "passenger_dest_y": 0, "passenger_loc_x": 0, "passenger_loc_y": 4, "x": 2, "y": 2}
            self.MAX_JOBS = 1
            self.fuel = 20
            self.x = 2
            self.y = 2
            self.jobs_done = 0
            self.passenger = False
            self.passenger_loc_x = 0
            self.passenger_loc_y = 4
            self.passenger_dest_x = 0
            self.passenger_dest_y = 0
            return self.__create_state()