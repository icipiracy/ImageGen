from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import ast
import gzip
import collections
import numpy
from six.moves import xrange
import cv2

from tensorflow.python.framework import dtypes

def split_data(data_index,data_dir,amount):

    """

        Split data into 'train' and 'test' sets.
        Data index = csv filename
        Data dir = directory in which data is stored
        Amount = percentage of data to be imported ( to reduce memory usage )

    """

    lfiles = []

    with open(data_index, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        lfiles = list(reader)

    dfiles = []
    dimages = []
    dlabels = []
    dlabel_names = []

    total_l = len(lfiles)
    amount = int(total_l / 100 * int(amount))

    print("Amount of dataset imported:")
    print(amount)

    for idx,val in enumerate(lfiles):

        if idx < amount:
            labels = ast.literal_eval(val[0])
            #labels = [n.strip() for n in x]

            labels_names = ast.literal_eval(val[2])
            #label_names = [n.strip() for n in y]

            dlabels.append(labels)

            # Fix file path to full path depending data_dir argv:
            dfiles.append(data_dir + val[1])

            dlabel_names.append(labels_names)

    ds_train = {
        'files':dfiles[:int(len(dfiles)/2)],
        'images':[],
        'labels':dlabels[:int(len(dlabels)/2)],
        'label_names':dlabel_names[:int(len(dlabel_names)/2)]
        }

    ds_test = {
        'files':dfiles[int(len(dfiles)/2):],
        'images':[],
        'labels':dlabels[int(len(dlabels)/2):],
        'label_names':dlabel_names[int(len(dlabel_names)/2):]
        }

    return [ds_train,ds_test]

def extract_saved_images(source):
    f = open(source, 'r')
    fnpy = numpy.load(f)
    return fnpy

def imageProcessor(source,saved=False,dataname="stat"):

    if saved:
        f = open(source, 'r')
        fnpy = numpy.load(f)
        raw_image_data = fnpy
    else:

        raw_image_data = []

        for idx,fpath in enumerate(source['files']):

            # img = numpy.array(Image.open(fpath).convert('RGBA'))
            # print(type(img))
            # print(img.shape)
            # img = np.array(img)
            img = cv2.imread(fpath)

            # shape = (800,800,3)
            # type = numpy.ndarray
            # value type = numpy.uint8

            # Remove 2 colors to keep black and white image:
            img = img[:,:,:1]

            raw_image_data.append(img)

        raw_image_data = numpy.array(raw_image_data)

        fname = 'raw_' + dataname + "_images"

        numpy.save('raw_image_data',raw_image_data)

    number_of_images = len(raw_image_data)
    width = raw_image_data[0].shape[0]
    height = raw_image_data[0].shape[1]

    print("\n Extract images with shape:")
    # No need to reshape, data is in right shape already:
    print(raw_image_data.shape) # (50, 800, 800, 1)

    return raw_image_data


def labelProcessor(source):

    raw_labels = []

    for idx,val in enumerate(source['labels']):
        val = numpy.array(val,dtype=numpy.float32)
        raw_labels.append(val)

    raw_labels = numpy.array(raw_labels)
    raw_labels.astype(numpy.float32)

    print("\n Extract labels with shape: ")
    print(raw_labels.shape)
    #print(type(raw_labels))
    #print(type(raw_labels[0]))

    return raw_labels


class DataSet(object):
    def __init__(self,images,labels,test=False,one_hot=False,dtype=dtypes.float32,reshape=True):

        dtype = dtypes.as_dtype(dtype).base_dtype

        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if test:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], ('images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            if reshape:
                assert images.shape[3] == 1

                images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])

            if dtype == dtypes.float32:
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):

        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0

            return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]

        start = self._index_in_epoch

        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]

        if start + batch_size > self._num_examples:

            self._epochs_completed += 1

            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)

                self._images = self.images[perm]
                self._labels = self.labels[perm]

                start = 0
                self._index_in_epoch = batch_size - rest_num_examples
                end = self._index_in_epoch
                images_new_part = self._images[start:end]
                labels_new_part = self._labels[start:end]
                return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def read_data_sets(data_dir,data_index,test,amount,saved=False,sfpath=False):

    print("\n\n")
    print("Read data sets with:")
    print("Data dir: %s" % data_dir)
    print("Data index: %s" % data_index)
    print("\n\n")

    one_hot=False
    dtype=dtypes.float32
    reshape=True

    spl = split_data(data_index,data_dir,amount)

    train = spl[0]
    test = spl[1]

    if saved and sfpath:

        train_images = imageProcessor(train,saved)
        train_labels = imageProcessor(train,saved)
        test_images = imageProcessor(test,saved)
        test_labels = imageProcessor(test,saved)
    else:
        train_images = imageProcessor(train)
        train_labels = labelProcessor(train)
        test_images = imageProcessor(test)
        test_labels = labelProcessor(test)

    validation_images = train_images[:int(train_images/3*1)]
    validation_labels = train_labels[:int(train_images/3*1)]
    train_images = train_images[int(train_images/3*1):]
    train_labels = train_labels[int(train_images/3*1):]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images,validation_labels,dtype=dtype,reshape=reshape)
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    dset = Datasets(train=train, validation=validation, test=test)

    result = {}
    result['dset'] = dset

    # Pass information to model to define new shape:
    imginf = {}
    imginf['width'] = train_images[0].shape[0]
    imginf['height'] = train_images[0].shape[1]
    imginf['depth'] = train_images[0].shape[2]
    imginf['classes'] = train_labels[0].shape[0]

    result['imginfo'] = imginf

    print(result['imginfo'])

    return result
