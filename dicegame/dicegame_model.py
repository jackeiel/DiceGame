from gym.spaces import Dict

from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch

from tensorflow.python.keras.layers import Input, Dense, Concatenate
from tensorflow.python.keras.models import Model

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

# https://github.com/ray-project/ray/blob/master/rllib/examples/models/action_mask_model.py


class DiceGameModel(TFModelV2):
    """Model that handles simple discrete action masking.

    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )

        super(DiceGameModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

        input1 = Input(shape=(10,), name='n_players_left')
        input2 = Input(shape=(50,), name='n_dice_left')
        input3 = Input(shape=(5,), name='n_dice')
        input4 = Input(shape=(5,), name='roll')
        input5 = Input(shape=(2,), name='ones_wild')

        concat_layer = Concatenate()([input1, input2, input3, input4, input5])
        dense1 = Dense(256, activation='relu')(concat_layer)
        dense2 = Dense(256, activation='relu')(dense1)
        dense3 = Dense(64, activation='relu')(dense2)

        action_out = Dense(301, activation=None, name='action_out')(dense3)
        value_out = Dense(1, activation=None, name='value_out')(dense3)
        internal_model = Model([input1, input2, input3, input4, input5], [action_out, value_out])
        self.internal_model = internal_model

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        x = input_dict["obs"]['observations']
        logits, self._value_out = self.internal_model(x)

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
