import copy
from rl_env import RamenFuryEnv, Action
from cards import Card

def test_play_card_in_own_bowl():
    n_players = 2
    hand_size = 5
    env = RamenFuryEnv(n_players, hand_size)
    player_number = 0
    card_idx = 1
    bowl = 2
    card_to_play = env.hands["player_0"][card_idx]
    cards_in_hand = len(env.hands["player_0"])
    env.step((Action.play_card.value, env.flatten_play_card(player_number, card_idx, bowl)))
    assert len(env.hands["player_0"]) < cards_in_hand
    assert env.hands["player_0"][card_idx] != card_to_play
    assert env.bowls[f"player_{player_number}"][bowl][-1] == card_to_play


def test_play_card_in_opponent_1_bowl():
    n_players = 2
    hand_size = 5
    env = RamenFuryEnv(n_players, hand_size)
    player_number = 1
    card_idx = 1
    bowl = 0
    card_to_play = env.hands["player_0"][card_idx]
    cards_in_hand = len(env.hands["player_0"])
    env.step((Action.play_card.value, env.flatten_play_card(player_number, card_idx, bowl)))
    assert len(env.hands["player_0"]) < cards_in_hand
    assert env.hands["player_0"][card_idx] != card_to_play  # a bit flaky but ok for now
    assert env.bowls[f"player_{player_number}"][bowl][-1] == card_to_play


def test_draw_card_from_pantry():
    n_players = 2
    hand_size = 5
    env = RamenFuryEnv(n_players, hand_size)
    pantry_idx = 2
    card_to_draw = env.pantry[pantry_idx]
    cards_in_hand = len(env.hands["player_0"])
    env.step((Action.draw_card.value, pantry_idx))
    assert len(env.hands["player_0"]) == cards_in_hand + 1
    assert env.hands["player_0"][-1] == card_to_draw
    assert env.pantry[pantry_idx] != card_to_draw  # bit flaky but ok
    for card in env.hands["player_0"]:
        assert isinstance(card, Card)

def test_draw_card_from_pantry_when_at_max():
    n_players = 2
    hand_size = 5
    env = RamenFuryEnv(n_players, hand_size)
    pantry_idx = 2
    while len(env.hands["player_0"]) < hand_size:  # draw until hand is full
        env.step((Action.draw_card.value, pantry_idx))
    
    card_to_draw = env.pantry[pantry_idx]
    cards_previously_in_hand = copy.deepcopy(env.hands["player_0"])
    env.step((Action.draw_card.value, pantry_idx))
    assert len(env.hands["player_0"]) == hand_size
    assert env.deck.discard_pile[-1] in cards_previously_in_hand
    

def test_eat_bowl():
    n_players = 2
    hand_size = 5
    env = RamenFuryEnv(n_players, hand_size)
    env.bowls["player_0"][0].append([1, 2])
    which_bowl_to_eat = 0
    obs, reward, done, info = env.step((Action.eat_bowl.value, which_bowl_to_eat))
    assert which_bowl_to_eat in env.eaten_bowls[f"player_0"]
    assert done == False
    for which_bowl_to_eat in [1, 2]:
        obs, reward, done, info = env.step((Action.eat_bowl.value, which_bowl_to_eat))
    assert done

def test_empty_bowl():
    n_players = 2
    hand_size = 5
    env = RamenFuryEnv(n_players, hand_size)
    player_number = 0
    card_idx = 1
    bowl = 2
    card_to_play = env.hands["player_0"][card_idx]
    cards_in_hand = len(env.hands["player_0"])
    # put two cards in the same bowl
    env.step((Action.play_card.value, env.flatten_play_card(player_number, card_idx, bowl)))
    env.step((Action.play_card.value, env.flatten_play_card(player_number, card_idx, bowl)))
    assert len(env.bowls[f"player_{player_number}"][bowl]) == 2
    env.step((Action.empty_bowl.value, bowl))
    assert len(env.deck.discard_pile) == 2

