from abc import ABCMeta
from abc import abstractmethod
from abc import abstractproperty
import os
import sys
import onnx_chainer
from chainer import functions as F

import chainer
from chainer import serializers
import numpy
import warnings


def load_npz_no_strict(filename, obj):
    try:
        serializers.load_npz(filename, obj)
    except KeyError as e:
        warnings.warn(repr(e))
        with numpy.load(filename) as f:
            d = serializers.NpzDeserializer(f, strict=False)
            d.load(obj)


class Agent(object, metaclass=ABCMeta):
    """Abstract agent class."""

    @abstractmethod
    def act_and_train(self, obs, reward):
        """Select an action for training.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def act(self, obs):
        """Select an action for evaluation.

        Returns:
            ~object: action
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode_and_train(self, state, reward, done=False):
        """Observe consequences and prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_episode(self):
        """Prepare for a new episode.

        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def save(self, dirname):
        """Save internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def load(self, dirname):
        """Load internal states.

        Returns:
            None
        """
        pass

    @abstractmethod
    def get_statistics(self):
        """Get statistics of the agent.

        Returns:
            List of two-item tuples. The first item in a tuple is a str that
            represents the name of item, while the second item is a value to be
            recorded.

            Example: [('average_loss': 0), ('average_value': 1), ...]
        """
        pass


class AttributeSavingMixin(object):
    """Mixin that provides save and load functionalities."""

    @abstractproperty
    def saved_attributes(self):
        """Specify attribute names to save or load as a tuple of str."""
        pass

    def save(self, dirname):
        """Save internal states."""
        self.__save(dirname, [])

    def __save(self, dirname, ancestors):
        os.makedirs(dirname, exist_ok=True)
        ancestors.append(self)
        for attr in self.saved_attributes:
            print("Starting iteration and Attr is: ")
            print(attr)
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            print("attr_valu is: ")
            print(attr_value)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor
                    for ancestor in ancestors
                ), "Avoid an infinite loop"
                print("saving outside of else, and attr_value is...")
                print(attr_value)
                attr_value.__save(os.path.join(dirname, attr), ancestors)
            else:
                print("In the ELSE and attr is: ")
                print(attr)
                save_path = os.path.join(dirname, '{}.npz'.format(attr))
                serializers.save_npz(save_path, getattr(self, attr))
                save_path = os.path.join(dirname, '{}.h5'.format(attr))
                serializers.save_hdf5(save_path, getattr(self, attr))
#                if (attr == "model"):
#                    numpy.set_printoptions(threshold=sys.maxsize)
#                    print("save_path is: ")
#                    print(save_path)
#                    print("Weights are: ")
#                    weights = numpy.load(save_path)
#                    for item in weights:
#                        print("\n Item" + item + ">>>>>>>>>>>>>>>>>>>>>>>>>")
#                        sz = str(weights[item].size)
#                        print("Size: " + sz)
#                        dim = str(weights[item].ndim)
#                        print("Dimensions: " + dim)
#                        print(weights[item])
#                        print(numpy.array(weights[item]))
                    # saving as universal network model...
                    # Prepare dummy data, not sure what the 4th value should be?
                model = getattr(self, attr)
                x = numpy.zeros((4, 84, 84), dtype=numpy.float32)[None]
                # Put Chainer into inference mode
                with chainer.using_config('train', False):
                    #chainer_out = model(x).array
                    chainer_out = model(x).q_values
                # Now save model
                onnx_model = onnx_chainer.export(getattr(self, attr), x, filename='convnet.onnx')
        ancestors.pop()

    def load(self, dirname):
        """Load internal states."""
        self.__load(dirname, [])

    def __load(self, dirname, ancestors):
        ancestors.append(self)
        for attr in self.saved_attributes:
            assert hasattr(self, attr)
            attr_value = getattr(self, attr)
            if attr_value is None:
                continue
            if isinstance(attr_value, AttributeSavingMixin):
                assert not any(
                    attr_value is ancestor
                    for ancestor in ancestors
                ), "Avoid an infinite loop"
                attr_value.load(os.path.join(dirname, attr))
            else:
                """Fix Chainer Issue #2772

                In Chainer v2, a (stateful) optimizer cannot be loaded from
                an npz saved before the first update.
                """
                load_npz_no_strict(
                    os.path.join(dirname, '{}.npz'.format(attr)),
                    getattr(self, attr))
        ancestors.pop()


class AsyncAgent(Agent, metaclass=ABCMeta):
    """Abstract asynchronous agent class."""

    @abstractproperty
    def process_idx(self):
        """Index of process as integer, 0 for the representative process."""
        pass

    @abstractproperty
    def shared_attributes(self):
        """Tuple of names of shared attributes."""
        pass


class BatchAgent(Agent, metaclass=ABCMeta):
    """Abstract agent class that can interact with a batch of envs."""

    @abstractmethod
    def batch_act(self, batch_obs):
        """Select a batch of actions for evaluation.

        Args:
            batch_obs (Sequence of ~object): Observations.

        Returns:
            Sequence of ~object: Actions.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_act_and_train(self, batch_obs):
        """Select a batch of actions for training.

        Args:
            batch_obs (Sequence of ~object): Observations.

        Returns:
            Sequence of ~object: Actions.
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset):
        """Observe a batch of action consequences for evaluation.

        Args:
            batch_obs (Sequence of ~object): Observations.
            batch_reward (Sequence of float): Rewards.
            batch_done (Sequence of boolean): Boolean values where True
                indicates the current state is terminal.
            batch_reset (Sequence of boolean): Boolean values where True
                indicates the current episode will be reset, even if the
                current state is not terminal.


        Returns:
            None
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_observe_and_train(
            self, batch_obs, batch_reward, batch_done, batch_reset):
        """Observe a batch of action consequences for training.

        Args:
            batch_obs (Sequence of ~object): Observations.
            batch_reward (Sequence of float): Rewards.
            batch_done (Sequence of boolean): Boolean values where True
                indicates the current state is terminal.
            batch_reset (Sequence of boolean): Boolean values where True
                indicates the current episode will be reset, even if the
                current state is not terminal.

        Returns:
            None
        """
        raise NotImplementedError()
