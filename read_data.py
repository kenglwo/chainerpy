
import numpy as np
from config import Config
from PIL import Image


def read_one_image(filename):
    # 画像を読み込んで･･･
    img = Image.open(filename)
    # モノクロ化（convert）
    if Config.IMAGE_MONO:
        img = img.convert('L')
    # リサイズ
    img = img.resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE), Image.ANTIALIAS)
    # np.array形式に変換
    img = np.array(img, dtype=np.float32)
    # (channel, height, width）の形式に変換
    if Config.IMAGE_MONO:
        img = img[np.newaxis, :]
    else:
        img = img.transpose([2, 0, 1])
    # 0-1のfloat値にする
    img /= 255.0
    return img


def read_data(filename):
    # ファイルを開く
    f = open(filename, 'r')
    # データを入れる配列
    images = []
    labels = []
    for line in f:
        # 改行を除いてスペース区切りにする
        line = line.rstrip()
        l = line.split()
        # イメージを読み込み
        images.append(read_one_image(l[0]))
        # 対応するラベルを用意
        labels.append(int(l[1]))
    f.close()

    # newImages = []
    # for image in images:
    #     newImages.append(np.array(image).astype(np.float32))
    #images = np.array(images, dtype=np.float32)
    #labels = np.array(labels, dtype=np.int32)
    #return images, labels
    return images, labels
