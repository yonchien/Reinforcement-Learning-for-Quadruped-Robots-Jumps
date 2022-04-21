import numpy as np
import pickle5 as pickle
from ray.rllib.utils.filter import Filter, MeanStdFilter


class PartialFilter(Filter):
    is_concurrent = False

    def __init__(self, shape):
        self.shape = shape

        load_filter = False
        if load_filter:
            with open('/home/zhuochen/DRCL/PyBulletTest/data_pickle/072721000518/sensor_filter.pkl', 'rb') as file:
                old_filter = pickle.load(file)
                self.sensor_data_filter = old_filter.sensor_data_filter
        else:
            self.sensor_data_filter = MeanStdFilter(shape=1800)
        self.sensor_seq_len = 30

    def __call__(self, x, update=True):
        x = np.asarray(x)
        sensor_data = x[:-12]
        prev_action = x[-12:]

        sensor_data = self.sensor_data_filter(sensor_data)

        # for i in range(self.sensor_seq_len):
        #     update = False
        #     if i == self.sensor_seq_len - 1:
        #         update = True
        #
        #     numerical = sensor_data[60 * i: 60 * (i + 1)]
        #     normalized = self.sensor_data_filter(numerical, update=update)
        #     sensor_data[60 * i: 60 * (i + 1)] = normalized

        return np.concatenate([sensor_data, prev_action])

    def apply_changes(self, other, *args, **kwargs):
        """Updates self with "new state" from other filter."""
        if "with_buffer" in kwargs:
            self.sensor_data_filter.apply_changes(other.sensor_data_filter, kwargs.get("with_buffer"))
        else:
            self.sensor_data_filter.apply_changes(other.sensor_data_filter)

    def copy(self):
        """Creates a new object with same state as self.
        Returns:
            A copy of self.
        """
        other = PartialFilter(self.shape)
        other.sync(self)
        return other

    def sync(self, other):
        """Copies all state from other filter to self."""
        assert other.shape == self.shape, "Shapes don't match!"
        self.sensor_data_filter.sync(other.sensor_data_filter)

    def clear_buffer(self):
        """Creates copy of current state and clears accumulated state"""
        self.sensor_data_filter.clear_buffer()

    def as_serializable(self):
        return self.copy()


class PartialFilter2(Filter):
    is_concurrent = False

    def __init__(self, shape):
        self.shape = shape

        load_filter = False
        if load_filter:
            with open('/home/zhuochen/DRCL/PyBulletTest/data_pickle/072721000518/sensor_filter.pkl', 'rb') as file:
                old_filter = pickle.load(file)
                self.sensor_data_filter = old_filter.sensor_data_filter
        else:
            self.sensor_data_filter = MeanStdFilter(shape=60)
        self.sensor_seq_len = 30

    def __call__(self, x, update=True):
        x = np.asarray(x)

        for i in range(self.sensor_seq_len):
            update = False
            if i == self.sensor_seq_len - 1:
                update = True

            numerical = x[64 * i: 64 * i + 60]
            normalized = self.sensor_data_filter(numerical, update=update)
            x[64 * i: 64 * i + 60] = normalized

        return x

    def apply_changes(self, other, *args, **kwargs):
        """Updates self with "new state" from other filter."""
        if "with_buffer" in kwargs:
            self.sensor_data_filter.apply_changes(other.sensor_data_filter, kwargs.get("with_buffer"))
        else:
            self.sensor_data_filter.apply_changes(other.sensor_data_filter)

    def copy(self):
        """Creates a new object with same state as self.
        Returns:
            A copy of self.
        """
        other = PartialFilter2(self.shape)
        other.sync(self)
        return other

    def sync(self, other):
        """Copies all state from other filter to self."""
        assert other.shape == self.shape, "Shapes don't match!"
        self.sensor_data_filter.sync(other.sensor_data_filter)

    def clear_buffer(self):
        """Creates copy of current state and clears accumulated state"""
        self.sensor_data_filter.clear_buffer()

    def as_serializable(self):
        return self.copy()
