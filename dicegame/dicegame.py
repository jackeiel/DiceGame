from gym.spaces import Box, Discrete, Dict
from gym.wrappers import FlattenObservation
from ray.rllib.env.multi_agent_env import MultiAgentEnv

import numpy as np


class DiceGame(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        self.randomize = config.get('randomize', False)
        self.max_players = 10
        self.n_players = config.get('n_players', 2)
        assert self.n_players <= self.max_players, 'Max number of players is 10'
        self.max_starting_dice = 5
        self.starting_dice_per_player = config.get('starting_dice_per_player', 2)
        assert self.starting_dice_per_player <= self.max_starting_dice, 'Max number of starting dice is 5'
        self.active_turn = 1
        self.ones_wild = 1
        self.prev_bid = 0
        self.n_dice_left = self.n_players * self.starting_dice_per_player
        self.player_obs = {}
        self._agent_ids = set(f'player_{i}' for i in range(1, self.max_players+1))

        self.observation_space = Dict({
            'observations': Dict({
                'n_players_left': Discrete(self.max_players),
                'n_dice_left': Discrete(self.max_players*self.max_starting_dice),
                'n_dice': Discrete(self.max_starting_dice),
                'roll': Box(shape=(self.max_starting_dice,), low=0, high=6, dtype=int),
                'ones_wild': Discrete(2)
            }),
            'action_mask': Box(0, 1, shape=(self.max_players*self.max_starting_dice*6+1,))
        })
        # total number of dice combinations, plus 1 for a call
        self.action_space = Discrete(self.max_players*self.max_starting_dice*6+1)

    def roll(self, n):
        roll = np.random.randint(low=1, high=6, size=n)
        zeros = np.zeros(self.max_starting_dice - n)
        roll = np.concatenate([roll, zeros])
        return roll

    def action_space_sample(self, agent_id_keys):
        agent_id = list(agent_id_keys)[0]
        return {agent_id: 1}  # sample bet of 1, 1

    # env needs a step, reset
    def reset(self):
        if self.randomize:
            # randomize starting position
            self.n_players = np.random.randint(low=2, high=self.max_players)
            self.starting_dice_per_player = np.random.randint(low=1, high=self.max_starting_dice)
        self.prev_bid = 0
        self.active_turn = 1  # always start on player 1 (why not?)
        self.ones_wild = 1  # True
        self.n_dice_left = self.n_players * self.starting_dice_per_player

        player_observation = {
                'n_players_left': self.n_players,
                'n_dice_left': self.n_dice_left,
                'n_dice': self.starting_dice_per_player,
                'ones_wild': 1
                }

        self.player_obs = {f'player_{i}': player_observation.copy() for i in range(1, self.n_players+1)}
        for p in self.player_obs:
            roll = self.roll(self.player_obs[p]['n_dice'])
            self.player_obs[p]['roll'] = roll

        player = f'player_{self.active_turn}'
        action_mask = self.get_action_mask(0)
        obs = self.get_obs(player, action_mask)
        return obs

    def step(self, action_dict):
        action = action_dict[f'player_{self.active_turn}']
        terminals = {'__all__': False}
        rewards = dict()

        if action == 0:  # call
            player = f'player_{self.active_turn}'
            last_bid = self.prev_bid
            # print('last bid was', last_bid)
            if self.check_bid(last_bid):  # Return True if bid is accurate, False if not
                # active player loses die
                # print('active turn', self.active_turn, 'player losing die', player)

                self.player_obs[player]['n_dice'] -= 1
                if self.player_obs[player]['n_dice'] == 0:
                    del self.player_obs[player]
                    terminals[player] = True
                    rewards[player] = -1
                next_player = self.set_next_player()  # returns a player string
                self.active_turn = int(next_player.split('_')[-1])
            else:
                # previous player loses die
                prev_player = self.find_prev_player()
                # print('active turn', self.active_turn, 'prev player losing die', prev_player)
                self.player_obs[prev_player]['n_dice'] -= 1
                if self.player_obs[prev_player]['n_dice'] == 0:
                    del self.player_obs[prev_player]
                    terminals[prev_player] = True
                    rewards[prev_player] = -1
                next_player = player
                self.active_turn = int(next_player.split('_')[-1])
            self.n_dice_left -= 1
            # determine if WIN
            if len(self.player_obs) == 1:
                p = list(self.player_obs.keys())[0]
                terminals[p] = True
                terminals['__all__'] = True
                rewards[p] = 1  # TODO we don't get all of the rewards at the end (missing the second player losing)

            # update player states
            players_left = len(self.player_obs)
            for p in self.player_obs:
                # if p not in terminals:
                #     terminals[p] = False
                self.player_obs[p]['n_players_left'] = players_left
                self.player_obs[p]['n_dice_left'] = self.n_dice_left
                self.player_obs[p]['roll'] = self.roll(self.player_obs[p]['n_dice'])
                self.player_obs[p]['ones_wild'] = 1  # after a call ones always reset to wild
        else:  # move turn along circle
            _, value = self.convert_int_to_bid(action)
            if value == 1:
                self.ones_wild = 0
            next_player = self.set_next_player()  # returns a player string
            self.active_turn = int(next_player.split('_')[-1])

        action_mask = self.get_action_mask(action)
        new_obs = self.get_obs(next_player, action_mask)

        infos = {}
        self.prev_bid = action
        return new_obs, rewards, terminals, infos

    def get_obs(self, player, action_mask):
        obs = {player: {
            'observations': self.player_obs[player],
            'action_mask': action_mask
        }}
        return obs

    def get_action_mask(self, prev_bet):
        action_mask = np.ones(shape=(self.max_players*self.max_starting_dice*6+1), dtype=int)
        action_mask[self.n_dice_left+1:] = 0
        if prev_bet == 0:
            action_mask[0] = 0
        else:
            action_mask[1: prev_bet+1] = 0
        return action_mask

    def check_bid(self, int_bid):
        """Return True if bid is accurate, False if not"""
        bid_n, bid_value = self.convert_int_to_bid(int_bid)
        actual_n = self.count_dice(bid_value)
        if bid_n > actual_n:
            return False
        else:
            return True

    def convert_int_to_bid(self, int_bid):
        if int_bid == 0:
            return 0, 0
        max_num = 6 * self.max_starting_dice * self.max_players
        dice = np.arange(1, max_num+1)
        dice = dice.reshape(self.max_players * self.max_starting_dice, 6)
        n, value = np.where(dice == int_bid)
        n = n[0] + 1
        value = value[0] + 1
        return n, value

    def count_dice(self, value):
        n = 0
        for player in self.player_obs:
            p = self.player_obs[player]
            if self.ones_wild:
                sub_n = len(list(filter(lambda i: i in [1, value], p['roll'])))
            else:
                sub_n = len(list(filter(lambda i: i == value, p['roll'])))
            n += sub_n
        return n

    def find_prev_player(self):
        players = self.player_obs.keys()
        players_sorted = sorted(players)
        turn = self.active_turn
        if turn == 1:  # player one's turn, just take the largest player left
            return players_sorted[-1]  # this WILL be previous player
        else:
            # work backwards
            for p in range(turn-1, 0, -1):
                if f'player_{p}' in players:
                    return f'player_{p}'
        # if we haven't found it yet, start again at top
        return players_sorted[-1]

    def set_next_player(self):
        turn = self.active_turn
        for p in range(turn+1, self.n_players+1):
            if f'player_{p}' in self.player_obs:
                self.active_turn = p
                return f'player_{p}'
        for p in range(1, turn):
            if f'player_{p}' in self.player_obs:
                self.active_turn = p
                return f'player_{p}'
