"""
Experiment with mlflow module
"""
import sys
sys.path.append('..')
import os
import numpy as np
from io import BytesIO
import gzip
import requests


def load_voting_records(filepath='./datasets/VOTING-RECORD/'):
    data_file = os.path.join(filepath, 'data.npz')
    if os.path.isfile(data_file):
        data = np.load(data_file)
        return data['X'], data['labels']

    print('Process voting record data')
    filename = 'house-votes-84.data'
    with open(os.path.join(filepath, filename), 'r') as f:
        lines = list(f)
        f.close()
    # maps y, n, ? to numerical values
    vote2num = {'y' : 1, 'n' : -1, '?' : 0}
    party2num = {'democrat': 1, 'republican': -1}
    X = [list(map(lambda x: vote2num[x], line.strip().split(',')[1:])) 
        for line in lines]
    X = np.array(X)
    labels = [party2num[line.split(',')[0]] for line in lines]
    labels = np.array(labels)
    np.savez(data_file, X=X, labels=labels) 
    return X, labels

def load_MNIST(params):
    """
    params:
        digits
        num_points
        seed
    """
    digits = params['digits']
    num_points = params['num_points']
    seed = params['seed']
    print("Loading the MNIST data with digits %s ..." % str(digits))
    if len(digits) != len(num_points):
        raise ValueError('Length of digits and num_points must be the same')

    filepath = "./datasets/MNIST/"
    filename = "all_digits.npz"

    if not os.path.exists(filepath):
        print('Folder for MNIST not already created, creating...')
        os.mkdir(filepath)

    if os.path.isfile(os.path.join(filepath, filename)):
        print('Found MNIST data already saved\n')
        mnist = np.load(os.path.join(filepath, filename))
        X, labels = mnist['X'], mnist['labels']
    else:
        print("Couldn't find already saved MNIST data for the given setup,\
               downloading from http://yann.lecun.com/exdb/mnist/")

        """ Loading data from MNIST website"""
        resp = requests.get(
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz').content
        file = BytesIO(resp)
        f = gzip.open(file, 'r')
        f.read(16)
        num_images = 60000  # read ALL the images
        img_size = 28
        buf = f.read(img_size * img_size * num_images)
        X = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X = X.reshape(num_images, img_size * img_size)

        resp = requests.get('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz').content
        file = BytesIO(resp)
        f = gzip.open(file, 'r')
        f.read(8)
        buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
        np.savez(os.path.join(filepath, filename), X=X, labels=labels)

    np.random.seed(seed)
    dig_ind = []
    for i in range(len(digits)):
        d_ind = np.where(labels == digits[i])[0]
        np.random.shuffle(d_ind)
        dig_ind.extend(list(d_ind[:num_points[i]]))

    X = X[dig_ind, :]
    labels = labels[dig_ind]
    return X, labels

if __name__ == '__main__':
    load_voting_records()
