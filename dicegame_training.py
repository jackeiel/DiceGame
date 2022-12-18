import os

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import PolicySpec

from dicegame.dicegame import DiceGame
from dicegame.dicegame_model import DiceGameModel

import numpy as np


def dg_training(config, path=''):
    ray.init()

    algo = config.build()

    # Can optionally call algo.restore(path) to load a checkpoint.
    for i in range(1000):
        print(f'######## TRAINING ITERATION {i} ########')
        # Perform one iteration of training the policy with PPO
        result = algo.train()

        if i % 100 == 0 or i in [2, 3, 10]:
            print("######## SAVING MODEL CHECKPOINT AND RUNNING A TRAINING GAME ########")
            if not os.path.exists(os.path.join(path, 'dice_models')):
                os.mkdir(os.path.join(path, 'dice_models'))
            checkpoint = algo.save(os.path.join(path, 'dice_models'))
            config = {
                'n_players': 3,
                'starting_dice_per_player': 3
            }
            game = DiceGame(config)
            obs = game.reset()
            if not os.path.exists(os.path.join(path, 'training_games')):
                os.mkdir(os.path.join(path, 'training_games'))
            f_name = os.path.join(path, 'training_games', f'training_game_{i}.txt')
            with open(f_name, 'w') as file:
                file.write(f'### START GAME {i}###')
                while True:
                    file.write('\nObservations:\n')
                    file.write(pretty_print(game.player_obs))
                    file.write('\nactive turn p')
                    file.write(str(game.active_turn))
                    action = algo.compute_single_action(obs[f'player_{game.active_turn}'], policy_id='learning_policy9')
                    file.write('\naction: ')
                    file.write(str(game.convert_int_to_bid(action)))
                    action_dict = {f'player_{game.active_turn}': action}
                    obs, reward, done, info = game.step(action_dict)
                    file.write('\nREWARD: ')
                    file.write(str(reward))
                    if done['__all__']:
                        break


if __name__ == "__main__":
    env_config = {
        'randomize': True
    }
    config = (
        ppo.PPOConfig()
        .resources(
            num_gpus=0
        )
        .rollouts(
            num_rollout_workers=2
        )
        .environment(
            DiceGame,
            env_config=env_config,
        )
        .multi_agent(
            policies={
                "learning_policy9": PolicySpec(
                    config={'gamma': 0.9}
                ),
                "learning_policy5": PolicySpec(
                    config={'gamma': 0.5}
                )
            },
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs:
            np.random.choice(['learning_policy5', 'learning_policy9']),
            policies_to_train=['learning_policy5', 'learning_policy9']
        )
        .training(
            model={
                "custom_model": DiceGameModel
            },
        )
        .framework('tf2')
    )

    dg_training(config=config)
