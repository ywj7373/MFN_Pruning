import pickle
import bcolz
import cv2
import mxnet as mx
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms as trans
import tfrecord
from pathlib import Path
from parser import args
import io
import tensorflow as tf
import os
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_tfrecords(rec_path):
    '''convert mxnet data to tfrecords.'''
    id2range = {}

    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'), str(rec_path / 'train.rec'), 'r')
    s = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(s)
    #print(header.label)
    imgidx = list(range(1, int(header.label[0])))
    seq_identity = range(int(header.label[0]), int(header.label[1]))
    for identity in seq_identity:
        s = imgrec.read_idx(identity)
        header, _ = mx.recordio.unpack(s)
        a, b = int(header.label[0]), int(header.label[1])
        id2range[identity] = (a, b)
    print('id2range', len(id2range))
    print('Number of examples in training set: {}'.format(imgidx[-1]))

    # generate tfrecords
    mx2tfrecords(imgidx, imgrec)


def mx2tfrecords(imgidx, imgrec):
    output_path = os.path.join(args.tfrecords_file_path, 'tran.tfrecords')
    if not os.path.exists(args.tfrecords_file_path):
        os.makedirs(args.tfrecords_file_path)
    writer = tf.python_io.TFRecordWriter(output_path)
    random.shuffle(imgidx)
    for i, index in enumerate(imgidx):
        img_info = imgrec.read_idx(index)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    print('%d num image processed' % i)
    writer.close()


def load_bin(path, rootdir, image_size=[112, 112]):
    transform = trans.Compose([
        trans.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        trans.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill([len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')
    for i in range(len(bins)):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data.shape)
    np.save(str(rootdir) + '_list', np.array(issame_list))
    return data, issame_list


def decode_image(features):
    train_transform = trans.Compose([
        trans.RandomHorizontalFlip(),
        trans.ToTensor(),
        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(io.BytesIO(features["image_raw"]))

    features["image_raw"] = train_transform(image)
    return features


def get_dataset(tfrecord_path, index_path):
    description = {"image_raw": "byte", "label": "int"}
    dataset = tfrecord.torch.TFRecordDataset(tfrecord_path, index_path=index_path, description=description,
                                             transform=decode_image)
    return dataset


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=path / name, mode='r')
    issame = np.load(path / '{}_list.npy'.format(name))
    return carray, issame


def get_val_data(data_path):
    agedb_30, agedb_30_issame = get_val_pair(data_path, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_val_pair(data_path, 'cfp_fp')
    lfw, lfw_issame = get_val_pair(data_path, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame


if __name__ == '__main__':
    data_path = Path(args.data_path)

    # Train dataset
    create_tfrecords(data_path)

    # Test dataset
    bin_files = ['agedb_30', 'cfp_fp', 'lfw']
    for i in range(len(bin_files)):
        load_bin(data_path / (bin_files[i] + '.bin'), data_path / bin_files[i])
