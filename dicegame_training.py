from pprint import pprint

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import PolicySpec

from dicegame.dicegame import DiceGame
from dicegame.dicegame_model import DiceGameModel

import numpy as np


def dg_training(config):
    ray.init()

    algo = config.build()

    # Can optionally call algo.restore(path) to load a checkpoint.
    for i in range(1000):
        print(f'######## TRAINING ITERATION {i} ########')
        # Perform one iteration of training the policy with PPO
        result = algo.train()
        # print(pretty_print(result))

        if i % 100 == 0 or i in [2, 3, 10]:
            print("######## SUPER SPECIAL RUN ########")
            checkpoint = algo.save('dice_models')
            print("checkpoint saved at", checkpoint)
            config = {
                'n_players': 3,
                'starting_dice_per_player': 3
            }
            game = DiceGame(config)
            obs = game.reset()
            with open(f'training_games/training_game_{i}.txt', 'w') as file:
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
                    # file.write('NEW OBS: ')
                    # file.write(str(obs[f'player_{game.active_turn}']['observations']))
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
