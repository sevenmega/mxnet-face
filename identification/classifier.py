#!/usr/bin/env python

import argparse
import logging
import os
import sys
import time

import cv2
import mxnet as mx
import numpy as np
import pandas as pd
import pickle
from operator import itemgetter

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.mixture import GMM
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from lightened_cnn import lightened_cnn_b_feature

ctx = mx.gpu(0)

def readimg(name, size, ctx):
    arr = np.zeros((1, 1, size, size), dtype=float)
    img = np.expand_dims(cv2.imread(name, 0), axis=0)
    arr[0][:] = img/255.0
    return arr

def get_rep(args, imgname):
    _, model_args, model_auxs = mx.model.load_checkpoint(args.model_prefix, args.epoch)
    symbol = lightened_cnn_b_feature()
    logging.info("processing {}".format(imgname))
    model_args['data'] = mx.nd.array(readimg(imgname, args.size, ctx), ctx)
    exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
    exector.forward(is_train=False)
    exector.outputs[0].wait_to_read()
    output = exector.outputs[0].asnumpy()
    #print output[0]
    #print output[0].shape
    #np.savetxt("output_savetxt.csv", output[0], delimiter=",") # in col?
    #output[0].tofile('output_tofile.csv',sep=',',format='%10.10f')
    return output

from data import iterImgs
import fileinput

def get_reps(args):
    imgs = list(iterImgs(args.aligned_prefix))
    print len(imgs)
    #reps = []
    reps_file = "{}/reps.csv".format(args.feature_prefix)
    label_file = "{}/labels.csv".format(args.feature_prefix)
    if os.path.isfile(reps_file):
        os.remove(reps_file)
    if os.path.isfile(label_file):
        os.remove(label_file)
    for img in imgs:
        print img.cls
        print img.label
        print img.name
        print img.path
        #reps.append(get_rep(args, img.path))
        get_rep(args, img.path).tofile('tmp.csv',sep=',',format='%10.10f')
        with open(reps_file, 'a') as fout:
            fin = fileinput.input('tmp.csv')
            for line in fin:
                fout.write(line)
                fout.write("\n")
            fin.close()
        with open(label_file, 'a') as fout:
            fout.write("{},{}\n".format(img.label, img.path));
    os.remove('tmp.csv')

def train(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.feature_prefix)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.feature_prefix)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)
    print("Training for {} classes.".format(nClasses))

    if args.classifier == 'LinearSvm':
        clf = SVC(C=1, kernel='linear', probability=True)
    elif args.classifier == 'GridSearchSvm':
        print("""
        Warning: In our experiences, using a grid search over SVM hyper-parameters only
        gives marginally better performance than a linear SVM with C=1 and
        is not worth the extra computations of performing a grid search.
        """)
        param_grid = [
            {'C': [1, 10, 100, 1000],
             'kernel': ['linear']},
            {'C': [1, 10, 100, 1000],
             'gamma': [0.001, 0.0001],
             'kernel': ['rbf']}
        ]
        clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
    elif args.classifier == 'GMM':  # Doesn't work best
        clf = GMM(n_components=nClasses)

    # ref:
    # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
    elif args.classifier == 'RadialSvm':  # Radial Basis Function kernel
        # works better with C = 1 and gamma = 2
        clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
    elif args.classifier == 'DecisionTree':  # Doesn't work best
        clf = DecisionTreeClassifier(max_depth=20)
    elif args.classifier == 'GaussianNB':
        clf = GaussianNB()

    # ref: https://jessesw.com/Deep-Learning/
    elif args.classifier == 'DBN':
        from nolearn.dbn import DBN
        clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                  learn_rates=0.3,
                  # Smaller steps mean a possibly more accurate result, but the
                  # training will take longer
                  learn_rate_decays=0.9,
                  # a factor the initial learning rate will be multiplied by
                  # after each iteration of the training
                  epochs=300,  # no of iternation
                  # dropouts = 0.25, # Express the percentage of nodes that
                  # will be randomly dropped as a decimal.
                  verbose=1)

    if args.ldaDim > 0:
        clf_final = clf
        clf = Pipeline([('lda', LDA(n_components=args.ldaDim)),
                        ('clf', clf_final)])

    clf.fit(embeddings, labelsNum)

    fName = "{}/classifier.pkl".format(args.classifier_prefix)
    print("Saving classifier to '{}'".format(fName))
    with open(fName, 'w') as f:
        pickle.dump((le, clf), f)

def infer(args, imgname):
    with open(args.classifier_model, 'r') as f:
        (le, clf) = pickle.load(f)

    rep = get_rep(args, imgname)
    start = time.time()
    predictions = clf.predict_proba(rep).ravel()
    maxI = np.argmax(predictions)
    person = le.inverse_transform(maxI)
    confidence = predictions[maxI]
    print("Prediction took {} seconds.".format(time.time() - start))
    print("Predict {} with {:.2f} confidence.".format(person, confidence))
    if isinstance(clf, GMM):
        dist = np.linalg.norm(rep - clf.means_[maxI])
        print("  + Distance from the mean: {}".format(dist))

# tp align
# openface/util/align-dlib.py ./raw align outerEyesAndNose ./aligned --size 128
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--singlerep', type=int, default=0,
                        help='Do single image rep')
    parser.add_argument('--feature', type=int, default=0,
                        help='Do feature extraction')
    parser.add_argument('--train', type=int, default=0,
                        help='Do train')
    parser.add_argument('--infer', type=int, default=0,
                        help='Do infer')
    parser.add_argument('--singleimg', type=str, default="./test.png",
                        help='Location of a single test image file')
    parser.add_argument('--suffix', type=str, default="png",
                        help='The type of image')
    parser.add_argument('--size', type=int, default=128,
                        help='the image size of lfw aligned image, only support squre size')
    parser.add_argument('--epoch', type=int, default=165,
                        help='The epoch number of model')
    parser.add_argument('--model-prefix', default='../model/lightened_cnn/lightened_cnn',
                        help='The trained model to get feature')
    parser.add_argument('--aligned-prefix', default='./aligned',
                        help='The aligned image dir')
    parser.add_argument('--feature-prefix', default='./feature',
                        help='The feature dir')
    parser.add_argument('--classifier-prefix', default='./classifier',
                        help='The classifier dir')
    parser.add_argument('--classifier-model', default='./classifier/classifier.pkl',
                        help='The classifier model')
    parser.add_argument('--ldaDim', type=int, default=-1)
    parser.add_argument(
        '--classifier',
        type=str,
        choices=[
            'LinearSvm',
            'GridSearchSvm',
            'GMM',
            'RadialSvm',
            'DecisionTree',
            'GaussianNB',
            'DBN'],
        help='The type of classifier to use.',
        default='LinearSvm')
    args = parser.parse_args()
    logging.info(args)
    if args.singlerep == 1:
        if not os.path.isfile(args.singleimg):
            logging.info("Test Image not present.")
            sys.exit(-1)
        get_rep(args, args.singleimg)
        sys.exit(0)
    if args.feature == 1:
        if not os.path.exists(args.aligned_prefix):
            logging.info("Error: aligned-prefix not present.")
            sys.exit(-1)
        if not os.path.exists(args.feature_prefix):
            os.makedirs(args.feature_prefix)
        get_reps(args)
        sys.exit(0)
    if args.train == 1:
        if not os.path.exists(args.feature_prefix):
            logging.info("Error: feature-prefix not present.")
            sys.exit(-1)
        if not os.path.exists(args.classifier_prefix):
            os.makedirs(args.classifier_prefix)
        train(args)
        sys.exit(0)
    if args.infer == 1:
        if not os.path.isfile(args.classifier_model):
            logging.info("Error: classifier-model not present.")
            sys.exit(-1)
        if not os.path.isfile(args.singleimg):
            logging.info("Test Image not present.")
            sys.exit(-1)
        infer(args, args.singleimg)
        sys.exit(0)

if __name__ == '__main__':
    main()
