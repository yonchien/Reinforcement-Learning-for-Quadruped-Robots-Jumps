import numpy as np

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import get_activation_fn, try_import_tf

# this is a newer version
tf1, tf, tfv = try_import_tf()
# tf = try_import_tf()

class LocomotionNetTF(TFModelV2):
    """Generic fully connected network implemented in ModelV2 API.
    Basically the same thing as rllib FCNet FullyConnectNetwork, but changing standard deviation
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        super(LocomotionNetTF, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        activation = get_activation_fn(model_config.get("fcnet_activation"))

        custom_model_config = model_config.get("custom_model_config")
        seq_len = custom_model_config.get("sensor_seq_len", 30)

        obs_len = int(np.product(obs_space.shape))

        inputs = tf.keras.layers.Input(
            shape=(obs_len,), name="observations")

        # inputs_sensor_data = inputs[..., :-1]
        # inputs_time_remain = inputs[..., -1]

        inputs_sensor_data = tf.keras.layers.Reshape(target_shape=(seq_len, int(obs_len / seq_len)))(inputs)

        sensor_lstm = tf.keras.layers.LSTM(units=128)(inputs_sensor_data)
        action_out = tf.keras.layers.Dense(
            num_outputs,
            name="fc_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(sensor_lstm)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(sensor_lstm)

        self.base_model = tf.keras.Model(
            inputs, [action_out, value_out])
        # self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs_flat"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def save_model(self, path):
        # self.base_model.save(path, save_format='tf')
        tf.saved_model.save(self.base_model, path)
