# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import numpy as np
# from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
# GAMERS = [mrugesh, abbey, quincy, kris]

def player(prev_play, opponent_history=[], my_history=[]):
    if not prev_play:
        opponent_history.clear()
    if prev_play:
        opponent_history.append(prev_play)
    game_space = ['R', 'P', 'S']
    game_dict = {'R': 1, 'P': 2, 'S': 3}
    start_act = ['R','R','R','P','S','S']
    ob_len = 3
    if len(opponent_history) < 6:
        guess = start_act[len(opponent_history)]
    else:
        if len(opponent_history) >= 6:
            # mrugesh RRPPPP
            # abbey   PPPPPP
            # quincy  RPPSRR
            # kris    PPPPSR
            if opponent_history[1] == 'R':
                judge_gamer = 0
            elif opponent_history[3] == 'S':
                judge_gamer = 2
            elif opponent_history[4] == 'S':
                judge_gamer = 3
            else:
                judge_gamer = 1
        Q_all = np.load('Q_all.npy')
        observation = [0,0]
        state = 0
        opp_state = 0
        my_state = 0
        for act in opponent_history[-ob_len:]:
            observation[0] *= 10
            observation[0] += game_dict[act]
        for act in my_history[-ob_len:]:
            observation[1] *= 10
            observation[1] += game_dict[act]
        for digit_index in range(ob_len):
            opp_state += (observation[0]//(10**digit_index)%10)*(3**digit_index)
            my_state += (observation[1]//(10**digit_index)%10)*(3**digit_index)
        state = opp_state * ((3**(ob_len+1)-1)//2) + my_state
        guess = game_space[np.argmax(Q_all[judge_gamer,state,:])]
    my_history.append(guess)
    return guess

# play(player,kris, 1000)