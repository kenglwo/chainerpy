import argparse

import numpy as np
from chainer import optimizers, serializers

from config import Config
from MyNet import MyNet
from read_data import read_data


def parse_arg():
    parser = argparse.ArgumentParser(description='Train convolution network.')
    parser.add_argument('-m', '--model', type=str,
                        help='specify a file name where trained model will be saved to.')
    parser.add_argument('-l', '--log_interval', type=int, default=10,
                        help='An interval of displaying a log.')
    parser.add_argument('train_data', type=str, nargs=1,
                        help='a training data where the file name of an image and its class are on each of the lines.')
    return parser.parse_args()


args = parse_arg()
train_images, train_labels = read_data(args.train_data[0])

model = MyNet()
optimizer = optimizers.Adam()
optimizer.setup(model)

num_train = len(train_images)

for epoch in range(100):
    accum_loss = None
    bs = Config.BATCH_SIZE
    perm = np.random.permutation(num_train)
    for i in range(0, num_train, bs):
        x_sample = np.array([train_images[idx] for idx in perm[i:(i+bs) if (i+bs < num_train) else num_train-1]])
#        x_sample = train_images[perm[i:(i + bs) if(i + bs < num_train) else num_train]]
        y_sample = np.array([train_labels[idx] for idx in perm[i:(i+bs) if (i+bs < num_train) else num_train-1]])

        model.zerograds()
        loss, acc = model.train(x_sample, y_sample)
        loss.backward()
        optimizer.update()

        accum_loss = loss if accum_loss is None else accum_loss + loss

    if epoch % args.log_interval == 0:
        print(epoch, accum_loss.data)

if args.model is not None:
    serializers.save_npz(args.model, model)
