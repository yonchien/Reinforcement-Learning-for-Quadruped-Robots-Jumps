import numpy as np

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from .transformer_layers import Encoder

# this is a newer version
tf1, tf, tfv = try_import_tf()


class TransEncoderNetTF(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API.
    Basically the same thing as rllib FCNet FullyConnectNetwork, but changing standard deviation
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(TransEncoderNetTF, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        activation = get_activation_fn(model_config.get("fcnet_activation"), "tanh")

        custom_model_config = model_config.get("custom_model_config")
        obs_seq_len = custom_model_config.get("sensor_seq_len", 30)
        is_training = custom_model_config.get("is_training", True)

        pe_max_len = obs_seq_len

        inputs = tf.keras.layers.Input(
            shape=(int(np.product(obs_space.shape)),), name="observations")

        inputs_sensor = tf.keras.layers.Reshape(target_shape=(obs_seq_len, 60))(inputs)

        encoder_out = Encoder(num_layers=2, d_model=64, num_heads=8, d_ff=128,
                              maximum_position_encoding=pe_max_len, rate=0)(
            inputs_sensor, training=is_training, mask=None)

        encoder_out = encoder_out[:, -1, :]

        # ff = tf.keras.layers.Dense(
        #     64,
        #     activation=activation,
        #     kernel_initializer=normc_initializer(1.0)
        # )(encoder_out)

        action_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(encoder_out)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(encoder_out)

        # self.base_model = tf.keras.Model(
        #     inputs, [transformer_out, attention_weights])
        self.base_model = tf.keras.Model(
            inputs, [action_out, value_out])
        # if not is_training:
        #     print(self.base_model.summary())
        # tf.keras.utils.plot_model(
        #     self.base_model, to_file='/home/zhuochen/transformer_model.png', show_shapes=False, show_dtype=False,
        #     show_layer_names=True, rankdir='LR', expand_nested=True, dpi=96,
        #     layer_range=None
        # )

    def forward(self, input_dict, state, seq_lens):
        # (model_out, self._value_out), _ = self.base_model(input_dict["obs_flat"])
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        # return model_out[:, -1, :], state
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def save_model(self, path):
        # self.base_model.save(path, save_format='tf')
        tf.saved_model.save(self.base_model, path)
