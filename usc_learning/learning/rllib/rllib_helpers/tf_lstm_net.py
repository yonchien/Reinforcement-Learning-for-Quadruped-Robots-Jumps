import numpy as np
import gym
from typing import List

from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.recurrent_net import RecurrentNetwork
from ray.rllib.models.tf.misc import normc_initializer

tf1, tf, tfv = try_import_tf()


class LSTMModel(RecurrentNetwork):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str, ):
        super(LSTMModel, self).__init__(obs_space, action_space, None, model_config, name)

        input_dim = int(np.product(self.obs_space.shape))
        self.cell_size = model_config["lstm_cell_size"]
        self.num_outputs = num_outputs

        input_layer = tf.keras.layers.Input(shape=(None, input_dim), name="inputs")
        state_in_h = tf.keras.layers.Input(shape=(self.cell_size,), name="h")
        state_in_c = tf.keras.layers.Input(shape=(self.cell_size,), name="c")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in", dtype=tf.int32)

        lstm_out, state_h, state_c = tf.keras.layers.LSTM(
            self.cell_size, return_sequences=True, return_state=True, name="lstm"
        )(
            inputs=input_layer,
            mask=tf.sequence_mask(seq_in),
            initial_state=[state_in_h, state_in_c],
        )

        action_out = tf.keras.layers.Dense(
            self.num_outputs,
            activation=None,
            name="fc_out",
            kernel_initializer=normc_initializer(0.01)
        )(lstm_out)
        values = tf.keras.layers.Dense(
            1,
            activation=None,
            name="values",
            kernel_initializer=normc_initializer(0.01))(
            lstm_out)

        self._rnn_model = tf.keras.Model(
            inputs=[input_layer, seq_in, state_in_h, state_in_c],
            outputs=[action_out, values, state_h, state_c],
        )

    @override(RecurrentNetwork)
    def forward_rnn(self, inputs: TensorType, state: List[TensorType], seq_lens: TensorType) -> (
            TensorType, List[TensorType]):
        model_out, self._value_out, h, c = self._rnn_model([inputs, seq_lens] + state)
        return model_out, [h, c]

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])

    def save_model(self, path):
        tf.saved_model.save(self._rnn_model, path)
