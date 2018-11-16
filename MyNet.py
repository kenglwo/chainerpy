from chainer import Chain
import chainer.links as L
import chainer.functions as F

from config import Config as C


class MySubNet(Chain):
    def __init__(self, in_ch, ch1, cv1, ch2, cv2):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_ch, ch1, cv1)
            self.conv2 = L.Convolution2D(ch1, ch2, cv2)

    def forward(self, x):
        h_1 = F.relu(self.conv1(x))
        h_2 = self.conv2(h_1)
        h_3 = F.max_pooling_2d(h_2, ksize=(4, 4))
        h_4 = F.local_response_normalization(h_3)
        return F.relu(h_4)


class MyNet(Chain):
    def __init__(self):
        img_channels = 1 if C.IMAGE_MONO else 3
        super().__init__()
        with self.init_scope():
            self.subNet = MySubNet(img_channels, C.CONV1_OUT_CHANNELS, C.CONV_SIZE, C.CONV2_OUT_CHANNELS, C.CONV_SIZE)
            self.fullLayer1 = L.Linear(C.NUM_HIDDEN_NEURONS1, C.NUM_HIDDEN_NEURONS2)
            self.fullLayer2 = L.Linear(C.NUM_HIDDEN_NEURONS2, C.NUM_CLASSES)

    def train(self, x, y_hat):
        y = self.forward(x)
        return F.softmax_cross_entropy(y, y_hat), F.accuracy(y, y_hat)

    def forward(self, x):
        h = self.subNet.forward(x)
        h = F.dropout(F.relu(self.fullLayer1(h)))
        return self.fullLayer2(h)
