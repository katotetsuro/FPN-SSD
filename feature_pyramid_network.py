import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers

from multibox import Multibox

from chainercv.links.model.ssd.ssd_vgg16 import _check_pretrained_model
from chainer import initializers

# copy from https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/ssd_vgg16.py#L24
_imagenet_mean = np.array((123, 117, 104)).reshape((-1, 1, 1))


class FeaturePyramidNetwork(chainer.Chain):
    insize = 300
    grids = (38, 19, 10, 5, 3, 1)

    def __init__(self, initialW=None):
        super().__init__()
        with self.init_scope():
            # bottom up
            self.resnet = ResNet50Layers(None)
            del self.resnet.fc6
            # top layer (reduce channel)
            self.toplayer = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0, initialW=initialW)

            # conv layer for top-down pathway
            self.conv_p4 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, initialW=initialW)
            self.conv_p3 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, initialW=initialW)
            self.conv_p2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, initialW=initialW)
            self.conv_p6 = L.Convolution2D(None, 256, ksize=3, stride=3, pad=2, initialW=initialW)
            self.conv_p7 = L.Convolution2D(None, 256, ksize=3, stride=3, pad=1, initialW=initialW)

            # lateral connection
            self.bn_p4 = L.BatchNormalization(256)
            self.lat_p4 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0, initialW=initialW)
            self.bn_p3 = L.BatchNormalization(256)
            self.lat_p3 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0, initialW=initialW)
            self.bn_p2 = L.BatchNormalization(256)
            self.lat_p2 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0, initialW=initialW)

    def __call__(self, x):
        # bottom-up pathway
        h = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
        h = F.max_pooling_2d(h, ksize=(2, 2))
        c2 = F.max_pooling_2d(self.resnet.res2(h), ksize=(2, 2))
        c3 = self.resnet.res3(c2)
        c4 = self.resnet.res4(c3)
        c5 = self.resnet.res5(c4)

        # top
        p5 = self.toplayer(c5)
        p4 = F.unpooling_2d(p5, ksize=2, outsize=(
                c4.shape[2:4])) + self.lat_p4(c4)
        p4 = self.bn_p4(p4)
        p4 = self.conv_p4(p4)

        p3 = F.unpooling_2d(p4, ksize=2, outsize=(
                c3.shape[2:4])) + self.lat_p3(c3)
        p3 = self.bn_p3(p3)
        p3 = self.conv_p3(p3)

        p2 = F.unpooling_2d(p3, ksize=2, outsize=(
                c2.shape[2:4])) + self.lat_p2(c2)
        p2 = self.bn_p2(p2)
        p2 = self.conv_p2(p2)

        # from paper,
        # Here we introduce P6 only for covering a larger anchor scale of 5122.
        # P6 is simply a stride two subsampling of P5
        # but original SSD prepare small( at spatial ) feature maps, so I try stride=3, it produece (4, 4) feature maps.
        p6 = self.conv_p6(p5)

        # adjust to SSD300(experimental)
        p7 = self.conv_p7(p6)

       # print('p2', F.max(p2))
       # print('p3', F.max(p3))
       # print('p4', F.max(p4))
       # print('p5', F.max(p5))
       # print('p6', F.max(p6))
       # print('p7', F.max(p7))

        return p2, p3, p4, p5, p6, p7


from chainercv.links.model.ssd import SSD


class FPNSSD(SSD):
    def __init__(self, n_fg_class=None, pretrained_model=None, init_scale=0.0001):
        # うまくいったらpretrained model配布しますね
        #        n_fg_class, path = _check_pretrained_model(
        #            n_fg_class, pretrained_model, self._models)

        print('initializing with scale={}'.format(init_scale))
        initializer = initializers.Normal(init_scale)
        super(FPNSSD, self).__init__(
            extractor=FeaturePyramidNetwork(initialW=initializer),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)),
                initialW=initializer),
            steps=(8, 16, 32, 64, 100, 300),
            sizes=(30, 60, 111, 162, 213, 264, 315),
            mean=_imagenet_mean)


#        if path:
#            _load_npz(path, self)
