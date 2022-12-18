from pprint import pprint

import ray
import ray.rllib.algorithms.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.examples.policy.random_policy import RandomPolicy

from dicegame import DiceGame
from dicegame_model import DiceGameModel

import numpy as np

def dg_training(config):
    ray.init()
    # config = ppo.DEFAULT_CONFIG.copy()
    # config["num_gpus"] = 0
    # config["num_workers"] = 1

    algo = config.build()

    # Can optionally call algo.restore(path) to load a checkpoint.
    for i in range(1000):
        print(f'######## TRAINING ITERATION {i} ########')
        # Perform one iteration of training the policy with PPO
        result = algo.train()
        print(pretty_print(result))

        if i % 100 == 0 or i in [2, 3, 10]:
            print("######## SUPER SPECIAL RUN ########")
            checkpoint = algo.save('dice_models')
            print("checkpoint saved at", checkpoint)
            config = {
                'n_players': 5,
                'starting_dice_per_player': 3
            }
            game = DiceGame(config)
            obs = game.reset()
            while True:
                print('active turn p', game.active_turn)
                action = algo.compute_single_action(obs[f'player_{game.active_turn}'], policy_id='learning_policy9')
                print('action:', game.convert_int_to_bid(action))
                action_dict = {f'player_{game.active_turn}': action}
                obs, reward, done, info = game.step(action_dict)
                print('NEW OBS: ')
                pprint(obs[f'player_{game.active_turn}']['observations'])
                print('REWARD: ', reward)
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
