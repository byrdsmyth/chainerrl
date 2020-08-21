import chainer
from chainer import functions as F
from chainer import links as L


class NatureDQNHead(chainer.ChainList):
    """DQN's head (Nature version)"""

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            #L.Convolution2D(n_input_channels, out_channel=32, ksize=8, stride=4, pad=0, nobias=False, initialW=None, initial_bias=bias, *, dilate=1, groups=1),
            L.Convolution2D(n_input_channels, 32, 8, stride=4,
                            initial_bias=bias),
            #L.Convolution2D(n_input_channels=32, out_channel=64, ksize=4, stride=2, pad=0, nobias=False, initialW=None, initial_bias=bias, *, dilate=1, groups=1),
            L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
            #L.Convolution2D(n_input_channels=64, out_channel=64, ksize=3, stride=1, pad=0, nobias=False, initialW=None, initial_bias=bias, *, dilate=1, groups=1),
            L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias),
            #L.Convolution2D(in_size=3136, out_size=n_output_channels, nobias=False, initialW=None, initial_bias=bias),
            L.Linear(3136, n_output_channels, initial_bias=bias),
#            L.Linear(7744, n_output_channels, initial_bias=bias),
        ]

        super(NatureDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h


class NIPSDQNHead(chainer.ChainList):
    """DQN's head (NIPS workshop version)"""

    def __init__(self, n_input_channels=4, n_output_channels=256,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 16, 8, stride=4,
                            initial_bias=bias),
            L.Convolution2D(16, 32, 4, stride=2, initial_bias=bias),
            L.Linear(2592, n_output_channels, initial_bias=bias),
        ]

        super(NIPSDQNHead, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h
