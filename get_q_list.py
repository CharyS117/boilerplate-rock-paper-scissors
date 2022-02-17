import numpy as np
import time
from RPS_game import mrugesh, abbey, quincy, kris
OB_LEN = 3
GAME_SPACE = ['R', 'P', 'S']
GAME_DICT = {'R': 1, 'P': 2, 'S': 3}
GAMERS = [mrugesh, abbey, quincy, kris]
START_ACT = [1,1,1,2,3,3]


class game:
    def __init__(self):
        self.state = 0
        self.observation = [0, int('1'*OB_LEN)]
        self.state_n = ((3**(OB_LEN+1)-1)//2)**2
        self.action_n = len(GAME_SPACE)
        self.result = [0, 0, 0]
        self.prev_action = ''

    def reset(self):
        self.state = 0
        self.observation = [0, 0]
        self.result = [0, 0, 0]
        self.prev_action = ''
        return self.state

    def step(self, opp_gamer, action):
        opp_action = opp_gamer(self.prev_action)
        self.observation[0] *= 10
        self.observation[0] += GAME_DICT[opp_action]
        self.observation[0] %= 10**OB_LEN
        self.observation[1] *= 10
        self.observation[1] += GAME_DICT[action]
        self.observation[1] %= 10**OB_LEN
        self.state = 0
        opp_state = 0
        my_state = 0
        for digit_index in range(OB_LEN):
            opp_state += (self.observation[0]//(10**digit_index)%10)*(3**digit_index)
            my_state += (self.observation[1]//(10**digit_index)%10)*(3**digit_index)
        self.state = opp_state * ((3**(OB_LEN+1)-1)//2) + my_state
        if action == opp_action:
            reward = -0.2
            self.result[2] += 1
        elif (action == 'R' and opp_action == 'P') or (action == 'P' and opp_action == 'S') or (action == 'S' and opp_action == 'R'):
            reward = -0.6
            self.result[1] += 1
        else:
            self.result[0] += 1
            reward = 1
        self.prev_action = action
        return self.state, reward


def get_q(env:game,gamer):
    Q = np.zeros((env.state_n, env.action_n))
    EPISODES = 150000  # how many times to run the enviornment from the beginning
    MAX_STEPS = 1000  # max number of steps allowed for each run of enviornment

    LEARNING_RATE = 0.87  # learning rate
    GAMMA = 0.96

    epsilon = 0.9
    gen_lim = 333
    prev100_win_times = np.array([gen_lim for _ in range(20)])
    prev_time_mark = 0
    for episode in range(EPISODES):
        state = env.reset()
        for step_index in range(MAX_STEPS):
            if step_index < len(START_ACT):
                action_index = START_ACT[step_index]-1
            elif np.random.uniform(0, 1) < epsilon:
                action_index = np.random.randint(env.action_n)
            else:
                action_index = np.argmax(Q[state, :])

            next_state, reward = env.step(gamer, GAME_SPACE[action_index])

            Q[state, action_index] += LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state, action_index])
            state = next_state
        time_mark = int(time.time()*1e3)
        if epsilon > 0.4 and env.result[0] > np.mean(prev100_win_times)+20:
            epsilon -= 0.002
            epsilon = round(epsilon,4)
        elif epsilon > 0.1 and env.result[0] > np.mean(prev100_win_times)+5:
            epsilon -= 0.0004
            epsilon = round(epsilon,5)
        elif env.result[0] > np.mean(prev100_win_times):
            epsilon -= 0.0001
            epsilon = round(epsilon,5)
            print(f'\rgamer={gamer.__name__},episode={episode},result={env.result},epsilon={epsilon},time/episode={time_mark-prev_time_mark}'+' '*20,end='')
        prev_time_mark = time_mark          
        if epsilon <= 0: break
        prev100_win_times[:-1] = prev100_win_times[1:]
        prev100_win_times[-1] = env.result[0]
    print(f'\rgamer={gamer.__name__} finished@episode={episode},result={env.result},epsilon={epsilon},time/episode={time_mark-prev_time_mark}'+' '*20)
    return Q

game_env = game()
Q_all = np.zeros((len(GAMERS),game_env.state_n, game_env.action_n))
for i in range(len(GAMERS)):
    Q_all[i,:,:] = get_q(game_env,GAMERS[i])
np.save('Q_all',Q_all)