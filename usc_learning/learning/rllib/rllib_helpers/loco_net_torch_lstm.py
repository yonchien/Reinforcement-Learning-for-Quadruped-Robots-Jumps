import numpy as np
from typing import Dict, List
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer, SlimFC, SlimConv2d
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


class PyTorchLocomotionModel(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(self, obs_space: gym.spaces.Space,
                 action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        custom_configs = model_config.get("custom_model_config")
        self._sensor_seq_len = custom_configs.get("sensor_seq_len", 30)

        activation = model_config.get("fcnet_activation", "tanh")

        self._sensor_lstm = nn.LSTM(60, 128, 1, batch_first=True)

        # self._all_fc1 = SlimFC(in_size=129,
        #                        out_size=100,
        #                        initializer=normc_initializer(1.0),
        #                        activation_fn=activation)
        #
        # self._all_fc2 = SlimFC(in_size=100,
        #                        out_size=100,
        #                        initializer=normc_initializer(1.0),
        #                        activation_fn=activation)

        self._action_layer = SlimFC(in_size=128,
                                    out_size=num_outputs,
                                    initializer=normc_initializer(0.01),
                                    activation_fn=None)
        self._value_layer = SlimFC(in_size=128,
                                   out_size=1,
                                   initializer=normc_initializer(0.01),
                                   activation_fn=None)

        self._features = None

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict["obs"].float()
        batch_size = obs.shape[0]

        # input_sensor = obs[:, :-1]
        # input_time_remain = obs[:, -1]
        # input_time_remain = torch.unsqueeze(input_time_remain, -1)

        input_sensor = torch.reshape(obs, (batch_size, self._sensor_seq_len, 60))

        output, (h, c) = self._sensor_lstm(input_sensor)

        # sensor_out = h[-1, ...]
        self._features = output[:, -1]

        # vision_out = torch.zeros(batch_size, 20).to(obs.device)

        # concat = torch.cat([sensor_out, input_time_remain], 1)

        # output = self._all_fc1(concat)
        # self._features = self._all_fc2(output)
        action_out = self._action_layer(self._features)

        return action_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        return self._value_layer(self._features).squeeze(1)

    def init_hidden(self, num_layers, batch_size, out_size, device):
        return (torch.zeros(num_layers, batch_size, out_size, device=device),
                torch.zeros(num_layers, batch_size, out_size, device=device))