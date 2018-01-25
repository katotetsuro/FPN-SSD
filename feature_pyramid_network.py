import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.links.model.vision.resnet import ResNet50Layers

from chainercv.links.model.ssd import Multibox

from chainercv.links.model.ssd.ssd_vgg16 import _check_pretrained_model

# copy from https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/ssd_vgg16.py#L24
_imagenet_mean = np.array((123, 117, 104)).reshape((-1, 1, 1))


class FeaturePyramidNetwork(chainer.Chain):
    insize = 300
    grids = (75, 38, 19, 10, 4)

    def __init__(self):
        super().__init__()
        with self.init_scope():
            # bottom up
            self.resnet = ResNet50Layers('auto')
            del self.resnet.fc6
            # top layer (reduce channel)
            self.toplayer = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)

            # conv layer for top-down pathway
            self.conv_p4 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv_p3 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)
            self.conv_p2 = L.Convolution2D(None, 256, ksize=3, stride=1, pad=1)

            # lateral connection
            self.lat_p4 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)
            self.lat_p3 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)
            self.lat_p2 = L.Convolution2D(
                in_channels=None, out_channels=256, ksize=1, stride=1, pad=0)

    def __call__(self, x):
        # bottom-up pathway
        h = F.relu(self.resnet.bn1(self.resnet.conv1(x)))
        h = F.max_pooling_2d(h, ksize=(2, 2))
        c2 = self.resnet.res2(h)
        c3 = self.resnet.res3(c2)
        c4 = self.resnet.res4(c3)
        c5 = self.resnet.res5(c4)

        # top
        p5 = self.toplayer(c5)
        p4 = self.conv_p4(
            F.unpooling_2d(p5, ksize=2, outsize=(
                c4.shape[2:4])) + self.lat_p4(c4))
        p3 = self.conv_p3(
            F.unpooling_2d(p4, ksize=2, outsize=(
                c3.shape[2:4])) + self.lat_p3(c3))
        p2 = self.conv_p2(
            F.unpooling_2d(p3, ksize=2, outsize=(
                c2.shape[2:4])) + self.lat_p2(c2))

        # from paper,
        # Here we introduce P6 only for covering a larger anchor scale of 5122.
        # P6 is simply a stride two subsampling of P5
        # but original SSD prepare small( at spatial ) feature maps, so I try stride=3, it produece (4, 4) feature maps.
        p6 = F.max_pooling_2d(p5, ksize=1, stride=3)

        return p2, p3, p4, p5, p6


from chainercv.links.model.ssd import SSD


class FPNSSD(SSD):
    def __init__(self, n_fg_class=None, pretrained_model=None):
        # うまくいったらpretrained model配布しますね
        #        n_fg_class, path = _check_pretrained_model(
        #            n_fg_class, pretrained_model, self._models)

        super(FPNSSD, self).__init__(
            extractor=FeaturePyramidNetwork(),
            multibox=Multibox(
                n_class=n_fg_class + 1,
                aspect_ratios=((2, 3), (2, 3), (2, 3), (2, 3), (2, 3))),
            steps=(8, 64, 100, 200, 300),
            sizes=(30, 60, 111, 162, 213, 315),
            mean=_imagenet_mean)


#        if path:
#            _load_npz(path, self)
