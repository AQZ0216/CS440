import numpy as np
import utils

class Agent:    
    def __init__(self, actions, Ne=40, C=40, gamma=0.7):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne # used in exploration function
        self.C = C
        self.gamma = gamma
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self,model_path):
        utils.save(model_path,self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self,model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
        s_prime = self.generate_state(environment)

        # TODO: write your function here
        if self._train:
            if self.s != None:
                self.N[self.s][self.a] += 1

                if dead:
                    reward = -1
                elif points > self.points:
                    reward = 1
                else:
                    reward = -0.1
                
                action = utils.RIGHT
                max_Q = self.Q[s_prime][utils.RIGHT]

                if self.Q[s_prime][utils.LEFT] > max_Q:
                    action = utils.LEFT
                    max_Q = self.Q[s_prime][utils.LEFT]
                elif self.Q[s_prime][utils.DOWN] > max_Q:
                    action = utils.DOWN
                    max_Q = self.Q[s_prime][utils.DOWN]
                elif self.Q[s_prime][utils.UP] > max_Q:
                    action = utils.UP
                    max_Q = self.Q[s_prime][utils.UP]

                self.Q[self.s][self.a] += self.C/(self.C + self.N[self.s][self.a]) * (reward + self.gamma * max_Q - self.Q[self.s][self.a])

            if self.N[s_prime][utils.RIGHT] < self.Ne:
                action = utils.RIGHT
            elif self.N[s_prime][utils.LEFT] < self.Ne:
                action = utils.LEFT
            elif self.N[s_prime][utils.DOWN] < self.Ne:
                action = utils.DOWN
            elif self.N[s_prime][utils.UP] < self.Ne:
                action = utils.UP
            else:
                action = utils.RIGHT
                max_Q = self.Q[s_prime][utils.RIGHT]
                if self.Q[s_prime][utils.LEFT] > max_Q:
                    action = utils.LEFT
                    max_Q = self.Q[s_prime][utils.LEFT]
                elif self.Q[s_prime][utils.DOWN] > max_Q:
                    action = utils.DOWN
                    max_Q = self.Q[s_prime][utils.DOWN]
                elif self.Q[s_prime][utils.UP] > max_Q:
                    action = utils.UP
                    max_Q = self.Q[s_prime][utils.UP]

            self.s = s_prime
            self.a = action
            self.points = points
        else:
            action = utils.RIGHT
            max_Q = self.Q[s_prime][utils.RIGHT]

            if self.Q[s_prime][utils.LEFT] > max_Q:
                action = utils.LEFT
                max_Q = self.Q[s_prime][utils.LEFT]
            elif self.Q[s_prime][utils.DOWN] > max_Q:
                action = utils.DOWN
                max_Q = self.Q[s_prime][utils.DOWN]
            elif self.Q[s_prime][utils.UP] > max_Q:
                action = utils.UP
                max_Q = self.Q[s_prime][utils.UP]

        if dead:
            self.reset()

        return action

    def generate_state(self, environment):
        # TODO: Implement this helper function that generates a state given an environment
        if environment[0] == environment[3]:
            food_dir_x = 0
        elif environment[0] > environment[3]:
            food_dir_x = 1
        else:
            food_dir_x = 2
        
        if environment[1] == environment[4]:
            food_dir_y = 0
        elif environment[1] > environment[4]:
            food_dir_y = 1
        else:
            food_dir_y = 2
        
        if environment[0] == 1:
            adjoining_wall_x = 1
        elif environment[0] == utils.DISPLAY_WIDTH-2:
            adjoining_wall_x = 2
        else:
            adjoining_wall_x = 0
        
        if environment[1] == 1:
            adjoining_wall_y = 1
        elif environment[1] == utils.DISPLAY_HEIGHT-2:
            adjoining_wall_y = 2
        else:
            adjoining_wall_y = 0

        adjoining_body_top = 0
        for i in environment[2]:
            if environment[0] == i[0] and environment[1]-1 == i[1]:
                adjoining_body_top = 1
                break
        
        adjoining_body_bottom = 0
        for i in environment[2]:
            if environment[0] == i[0] and environment[1]+1 == i[1]:
                adjoining_body_bottom = 1
                break
        
        adjoining_body_left = 0
        for i in environment[2]:
            if environment[0]-1 == i[0] and environment[1] == i[1]:
                adjoining_body_left = 1
                break

        adjoining_body_right = 0
        for i in environment[2]:
            if environment[0]+1 == i[0] and environment[1] == i[1]:
                adjoining_body_right = 1
                break
        
        return (food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right)