import sys
import argparse

import numpy as np
import chainer
from chainer import serializers

from MyNet import MyNet
from read_data import read_one_image


def parse_arg():
    parser = argparse.ArgumentParser(description='Test the trained convolution network.')
    parser.add_argument('-m', '--model', type=str, nargs=1,
                        help='specify a file name of a trained model.')
    parser.add_argument('test_images', type=str, nargs='+',
                        help='file name(s) of test image(s).')
    return parser.parse_args()


args = parse_arg()

model = MyNet()
serializers.load_npz(args.model[0], model)

test_images = []
for img in args.test_images:
    test_images.append(read_one_image(img))

for i in range(len(test_images)):
    img = np.array([test_images[i]], dtype=np.float32)
    with chainer.using_config("train", False):
        y = model.forward(img)
        pred = np.argmax(y)
        print(pred)
