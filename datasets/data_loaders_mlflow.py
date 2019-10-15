"""
Experiment with mlflow module
"""
import sys
sys.path.append('..')
import os
import mlflow
import tempfile
import numpy as np
from util.mlflow_util import get_prev_run

def load_voting_records(filepath='./datasets/VOTING-RECORD/', debug=False):
    if not debug:
        prev_run = get_prev_run(
            function    = 'data_loaders_mlflow.load_voting_records', 
            params      = {}, 
            git_commit  = None)
        if prev_run is not None:
            print("Found previously processed voting record data")
            data_uri = os.path.join(
                prev_run.info.artifact_uri,
                'data.npz')
            return data_uri
    
    print('Process voting record data')
    filename = 'house-votes-84.data'
    with open(filepath + filename, 'r') as f:
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
    data_file = os.path.join(filepath, 'data.npz')
    np.savez(data_file, X=X, labels=labels) 

    if debug:
        return data_file

    with mlflow.start_run(nested=True):
        mlflow.set_tag('function', 'data_loaders_mlflow.load_voting_records')
        mlflow.log_artifact(data_file)
        os.remove(data_file)
        data_uri = mlflow.get_artifact_uri()
        return os.path.join(data_uri, 'data.npz')


def load_MNIST(params):
    """
    params:
        digits:
        num_points
    """
    digits = params['digits']
    num_points = params['num_points']
    print("Loading the MNIST data with digits %s ..." % str(digits))
    if len(digits) != len(num_points):
        raise ValueError('Length of digits and num_points must be the same')

    # filepath and filename creation for checking if this data has already been computed before
    filename = "".join([str(d)+"_" for d in digits])
    filename += "".join([str(n)+"_" for n in num_points])
    filename += "%d_%s_%d.npz" % (num_eig, Ltype, k_nn)
    file_path = "./datasets/MNIST/"

    if not os.path.exists(file_path):
        print('Folder for MNIST not already created, creating...')
        os.mkdir(file_path)

    if os.path.isfile(file_path + filename):
        print('Found MNIST data already saved\n')
        mnist = np.load(file_path+filename)
        X, evals, evecs, ground_truth = mnist['X'], mnist['evals'], mnist['evecs'], mnist['ground_truth']
        np.random.seed(seed) # set the random seed to allow for consistency in choosing fidelity point
        fid = {}
        for i in np.unique(ground_truth).astype(int):
            i_ind = np.where(ground_truth == i)[0]
            np.random.shuffle(i_ind)
            fid[i] = list(i_ind[:int(sup_percent*num_points[i])])

        del mnist

    else:
        print("Couldn't find already saved MNIST data for the given setup, downloading from http://yann.lecun.com/exdb/mnist/")

        """ Loading data from MNIST website"""
        resp = requests.get('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz').content
        file = BytesIO(resp)
        f = gzip.open(file, 'r')
        f.read(16)
        num_images = 60000  # read ALL the images
        img_size = 28
        buf = f.read(img_size * img_size * num_images)
        imgs = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        imgs = imgs.reshape(num_images, img_size * img_size)

        resp = requests.get('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz').content
        file = BytesIO(resp)
        f = gzip.open(file, 'r')
        f.read(8)
        buf = f.read(num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)

        """ Processing to get the desired subset of digits"""
        dig_ind = []
        for j in range(len(digits)):
            d_ind = np.where(labels == digits[j])[0]
            np.random.shuffle(d_ind)
            dig_ind.extend(list(d_ind[:num_points[j]]))


        """ Define fid, ground_truth, and X datums"""
        X = imgs[dig_ind, :]
        labels_sub = labels[dig_ind]
        ground_truth = np.zeros(len(dig_ind))
        fid = {}
        if len(digits) == 2: # Binary case -- make the classes -1, 1
            m1_ind = np.where(labels_sub == digits[0])[0]
            ground_truth[m1_ind] = -1
            np.random.shuffle(m1_ind)
            fid[-1] = list(m1_ind[:int(sup_percent*num_points[0])])

            p1_ind = np.where(labels_sub == digits[1])[0]
            ground_truth[p1_ind] = 1
            np.random.shuffle(p1_ind)
            fid[1] = list(p1_ind[:int(sup_percent*num_points[1])])

        else:
            for i in range(len(digits)):
                i_ind = np.where(labels_sub == digits[i])[0]
                ground_truth[i_ind] = i
                np.random.shuffle(i_ind)
                fid[i] = list(i_ind[:int(sup_percent*num_points[i])])

        """ Create the similarity graph and calculate eigenval/vecs """
        W = make_sim_graph(X, k_nn=k_nn)
        if Ltype == 'n':
            evals, evecs = get_eig_Lnorm(W, num_eig=num_eig, normed_=True)
        else:
            evals, evecs = get_eig_Lnorm(W, num_eig=num_eig, normed_=False)

        # whether or not we will be keeping track of the FULL dataset
        if not full:
            X = None

        print('Saving MNIST data to %s' % (file_path+filename))
        # NOT SAVING fid, since will allow for different runs on fidelity
        np.savez(file_path + filename, X=X, evals=evals, evecs=evecs, ground_truth=ground_truth)

    # Reset the seed for random choices to be random, so as to allow for different runs
    np.random.seed(None)
    return Data_obj(X, evals, evecs, fid, ground_truth)

if __name__ == '__main__':
    load_voting_records()
