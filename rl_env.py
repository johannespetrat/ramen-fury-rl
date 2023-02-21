from enum import Enum
import gym
import copy
from gym import spaces
import numpy as np
from cards import Deck, Type, Card
from typing import List
from collections import Counter

ILLEGAL_ACTION_PENALTY = 0.

class Action(Enum):
    play_card = 1
    draw_card = 2
    eat_bowl = 3
    empty_bowl = 4

class RamenFuryEnv(gym.Env):
    def __init__(self, n_players=2, hand_size=5):
        # self.observation_space = spaces.Dict(
        #         {
        #             "pantry": spaces.MultiDiscrete([len(Type)] * 5),
        #             "hand": spaces.MultiDiscrete([len(Type)] * hand_size),
        #             "own_bowl_0": spaces.MultiDiscrete([len(Type)] * 5),
        #             "own_bowl_1": spaces.MultiDiscrete([len(Type)] * 5),
        #             "own_bowl_2": spaces.MultiDiscrete([len(Type)] * 5),
        #             # "opponent_bowls": spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         }
        #     )
        # 5 in pantry, and 5 for each bowl
        self.observation_space = spaces.MultiDiscrete([len(Type)] * (5 + hand_size + 15 * n_players))
        self.hand_size = hand_size
        self.n_players = n_players
        self.action_space = spaces.MultiDiscrete([len(Action), n_players * hand_size * 3])
        self.deck = Deck()
        self.set_up_game()

    def get_possible_actions(self, n_cards_in_hand, player_number):
        play_card = []
        for player_number in range(self.n_players):
            for card_idx in range(len(self.hands[f"player_{player_number}"])):
                for bowl in range(3):
                    if len(self.bowls[f"player_{player_number}"][bowl]) < 5:
                        play_card.append([1, self.flatten_play_card(player_number, card_idx, bowl)])
        draw_card = [[2, i] for i in range(len(self.pantry) + 1)] # five pantry piles and one from deck
        eat_bowl = [[3, i] for i in range(3)]
        empty_bowl = [[4, i] for i in range(3)]
        for bowl in self.eaten_bowls[f"player_{player_number}"]:
            if [1, bowl] in play_card:
                play_card.remove([1, bowl])
            eat_bowl.remove([3, bowl])
            empty_bowl.remove([4, bowl])
        return play_card + draw_card + eat_bowl + empty_bowl

    def check_action_legal(self, action, n_cards_in_hand, player_number):
        return list(action) in self.get_possible_actions(n_cards_in_hand, player_number)

    def reset(self):
        self.set_up_game()
        return self.game_state_to_observation(player_number=0)

    def score_bowl(self, bowl: List[Card]):
        broths = [card for card in bowl if card.is_broth()]
        if (len(broths) > 1) or (len(broths) == 0):
            return 0
        if len(bowl) > 5:
            return 0
        broth = broths[0]
        proteins = [card.value for card in bowl if card.is_protein()]
        veggies = [card.value for card in bowl if card.is_vegetarian()]
        n_chilies = len([card for card in bowl if card.is_chili()])
        n_noris = len([card for card in bowl if card.is_nori()])
        if broth.value == Type.fury.value:
            return 2 * n_chilies + n_noris
        if broth.value == Type.beef.value:
            n_unique = len(np.unique(proteins))
            return np.array([2, 5, 9, 14])[n_unique - 1] + n_noris - n_chilies
        if broth.value == Type.soy_sauce.value:
            n_unique = len(np.unique(veggies))
            return np.array([2, 5, 9, 14])[n_unique - 1] + n_noris - n_chilies
        if broth.value == Type.shrimp.value:
            n_tofu = len([card for card in bowl if card.value == Type.both.value])
            return max([(min([len(veggies), len(proteins)]) - n_tofu), 0]) * 4 + n_noris - n_chilies
        if broth.value == Type.chicken.value:
            counts = Counter([card.value for card in bowl if card.is_protein() or card.is_vegetarian()])
            if len(counts) == 0:
                return 0
            max_count = max([count for _, count in counts.items()])
            if max_count == 2:
                return 6 + n_noris - n_chilies
            elif max_count >= 3:
                return 10 + n_noris - n_chilies
            else:
                return 0

    
    def game_state_to_observation(self, player_number):
        player_hand = [card.value for card in self.hands[f"player_{player_number}"]]
        if len(player_hand) < self.hand_size:
            player_hand += [Type.no_card.value] * (self.hand_size - len(player_hand))
        bowls = self.bowls[f"player_{player_number}"]

        state_dict = {
            "pantry": np.array([card.value for card in self.pantry], dtype=np.float32),
            "hand": np.array(player_hand, dtype=np.float32),
            "own_bowl_0": np.array([c.value for c in bowls[0]] + [Type.no_card.value for _ in range(5-len(bowls[0]))]),
            "own_bowl_1": np.array([c.value for c in bowls[1]] + [Type.no_card.value for _ in range(5-len(bowls[1]))]),
            "own_bowl_2": np.array([c.value for c in bowls[2]] + [Type.no_card.value for _ in range(5-len(bowls[2]))])
            }
        keys = sorted(state_dict.keys())
        return np.concatenate([state_dict[key] for key in keys])

    def set_up_game(self):
        self.deck = Deck()
        self.hands = {f"player_{i}": self.deck.draw_card(n=3) for i in range(self.n_players)}
        self.pantry = self.deck.draw_card(n=5)
        self.bowls = {f"player_{i}": [[], [], []] for i in range(self.n_players)}
        self.eaten_bowls = {f"player_{i}": set([]) for i in range(self.n_players)}

    def _unflatten_play_card(self, number):
        actions_as_arr = np.arange(self.n_players * self.hand_size * 3).reshape(self.n_players, self.hand_size, 3)
        actions_as_matrix = actions_as_arr.reshape(self.n_players, self.hand_size, 3)
        player_number, card_idx, bowl = [i[0] for i in np.where(actions_as_matrix == number)]
        return player_number, card_idx, bowl

    def flatten_play_card(self, player_number, card_idx, bowl):
        actions_as_arr = np.arange(self.n_players * self.hand_size * 3).reshape(self.n_players, self.hand_size, 3)
        actions_as_matrix = actions_as_arr.reshape(self.n_players, self.hand_size, 3)
        return actions_as_matrix[player_number, card_idx, bowl]

    def step(self, action):
        is_action_legal = self.check_action_legal(action=action, n_cards_in_hand=len(self.hands["player_0"]), player_number=0)
        done = False
        reward = 0
        info = {}
        if is_action_legal:
            if action[0] == 1:  # play card
                player_number, card_idx, bowl = self._unflatten_play_card(action[1])
                self.bowls[f"player_{player_number}"][bowl].append(self.hands[f"player_0"][card_idx])
                del self.hands[f"player_0"][card_idx]
            elif action[0] == 2:  # pick from pantry
                idx = action[1]
                if idx == len(self.pantry):
                    self.hands["player_0"].append(self.deck.draw_card(1)[0])
                else:
                    card_in_pantry = copy.deepcopy(self.pantry[idx])
                    self.hands["player_0"].append(card_in_pantry)
                    self.pantry[idx] = self.deck.draw_card(1)[0]
                if len(self.hands["player_0"]) > self.hand_size:
                    cards_to_discard = [self.hands["player_0"].pop(0)]
                    self.deck.add_to_discard_pile(cards_to_discard)
            elif action[0] == 3:  # eat bowl
                which_bowl_to_eat = action[1]
                reward = self.score_bowl(self.bowls[f"player_0"][which_bowl_to_eat])
                self.eaten_bowls[f"player_0"].add(which_bowl_to_eat)
                done = len(self.eaten_bowls[f"player_0"]) == 3
            elif action[0] == 4:  # empty bowl
                which_bowl_to_empty = action[1]
                cards_to_discard = self.bowls[f"player_0"][which_bowl_to_empty]  # player 0
                self.deck.add_to_discard_pile(cards_to_discard)
                self.bowls["player_0"][which_bowl_to_empty] = []
                reward = 0
                done = False
            info = {}
        else:
            obs = self.game_state_to_observation(player_number=0)
            reward = ILLEGAL_ACTION_PENALTY
            done = False
            info = {"msg": "took illegal action"}
        obs = self.game_state_to_observation(0)
        if len(self.deck) == 0 and len(self.deck.discard_pile) == 0:
            done = True
        return obs, reward, done, info


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.env_util import make_vec_env
    from gym.wrappers import FlattenObservation
    env = RamenFuryEnv(1, 5)
    check_env(env)

    from stable_baselines3 import PPO
    ent_coefs = [0.01, 0., 0.001, 0.1]  # ent_coef = 0.01
    n_runs = 3
    n_iter = 1_000_000
    policy = "MlpPolicy"
    for ent_coef in ent_coefs:
        for _ in range(n_runs):
            vec_env = make_vec_env(RamenFuryEnv, n_envs=4, env_kwargs={"n_players": 1, "hand_size": 5})
            # model = PPO("MultiInputPolicy", vec_env, verbose=1, ent_coef=ent_coef, tensorboard_log="logs/")
            model = PPO("MlpPolicy", vec_env, verbose=1, ent_coef=ent_coef, tensorboard_log="logs/")
            model.learn(total_timesteps=n_iter, tb_log_name=f"PPO_n_iter_{n_iter}_ef_{ent_coef}_{policy}")
            # model.save("ramen-fury-ppo")