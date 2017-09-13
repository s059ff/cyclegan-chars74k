import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):

    def __init__(self):
        super(Generator, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            # (De)Convolution2D(input channels, output channels, kernel size, stride, padding)
            self.c0 = L.Convolution2D(1, 10, ksize=4, stride=2, pad=1, initialW=w)
            self.c1 = L.Convolution2D(10, 20, ksize=4, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution2D(20, 40, ksize=4, stride=2, pad=1, initialW=w)
            self.dc0 = L.Deconvolution2D(40, 20, ksize=4, stride=2, pad=1, initialW=w)
            self.dc1 = L.Deconvolution2D(20, 10, ksize=4, stride=2, pad=1, initialW=w)
            self.dc2 = L.Deconvolution2D(10, 1, ksize=4, stride=2, pad=1, initialW=w)
            self.bn_c0 = L.BatchNormalization(10)
            self.bn_c1 = L.BatchNormalization(20)
            self.bn_c2 = L.BatchNormalization(40)
            self.bn_dc0 = L.BatchNormalization(20)
            self.bn_dc1 = L.BatchNormalization(10)
            self.bn_dc2 = None      # Don't use batch normalization in output layer!

    def __call__(self, x):
        h = F.relu(self.bn_c0(self.c0(x)))
        h = F.relu(self.bn_c1(self.c1(h)))
        h = F.relu(self.bn_c2(self.c2(h)))
        h = F.relu(self.bn_dc0(self.dc0(h)))
        h = F.relu(self.bn_dc1(self.dc1(h)))
        h = F.tanh(self.dc2(h))
        return h

class Discriminator(chainer.Chain):

    def __init__(self):
        super(Discriminator, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(1, 10, ksize=4, stride=2, pad=1, initialW=w)
            self.c1 = L.Convolution2D(10, 20, ksize=4, stride=2, pad=1, initialW=w)
            self.c2 = L.Convolution2D(20, 40, ksize=4, stride=2, pad=1, initialW=w)
            self.c3 = L.Convolution2D(40, 1, ksize=3, stride=1, pad=1, initialW=w)
            self.bn0 = None      # Don't use batch normalization in input layer!
            self.bn1 = L.BatchNormalization(20)
            self.bn2 = L.BatchNormalization(40)

    def __call__(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = self.c3(h)
        return h
