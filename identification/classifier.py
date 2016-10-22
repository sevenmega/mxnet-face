#!/usr/bin/env python

import argparse
import logging
import os
import sys
import time
import fileinput

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
from data import iterImgs
from align_dlib import AlignDlib
import msgpack as mp

ctx = mx.gpu(0)

def readimg(imgname, size):
    logging.info("reading {}".format(imgname))
    arr = np.zeros((1, 1, size, size), dtype=float)
    img = np.expand_dims(cv2.imread(imgname, 0), axis=0)
    # print("img shape {}".format(img.shape))
    # print img
    arr[0][:] = img/255.0
    return arr

def get_rep(args, imgarr, time_measure=False, time_iter=10):
    _, model_args, model_auxs = mx.model.load_checkpoint(args.model_prefix, args.epoch)
    symbol = lightened_cnn_b_feature()
    model_args['data'] = mx.nd.array(imgarr, ctx)
    # print("input shape {}".format(imgarr.shape))
    # print imgarr
    exector = symbol.bind(ctx, model_args ,args_grad=None, grad_req="null", aux_states=model_auxs)
    exector.forward(is_train=False)
    exector.outputs[0].wait_to_read()
    output = exector.outputs[0].asnumpy()
    # print("output shape {}".format(output.shape))
    # print output[0]
    if time_measure:
        start = time.time()
        for index in range(time_iter):
            exector.forward(is_train=False)
            exector.outputs[0].wait_to_read()
        print("forward {} times took {} seconds, avg {}.".format(time_iter, time.time() - start, (time.time() - start)/time_iter))
    return output

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
        get_rep(args, readimg(img.path, args.size)).tofile('tmp.csv',sep=',',format='%10.10f')
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

def encode(obj):
    """
    Data encoder for serializing numpy data types.
    """
    if isinstance(obj, np.ndarray):
        return {b'nd': True,
                b'type': obj.dtype.str,
                b'shape': obj.shape,
                b'data': obj.tostring()}
    elif isinstance(obj, (np.bool_, np.number)):
        return {b'nd': False,
                b'type': obj.dtype.str,
                b'data': obj.tostring()}
    elif isinstance(obj, complex):
        return {b'complex': True,
                b'data': obj.__repr__()}
    else:
        return obj

def decode(obj):
    """
    Decoder for deserializing numpy data types.
    """

    try:
        if b'nd' in obj:
            if obj[b'nd'] is True:
                return np.fromstring(obj[b'data'],
                            dtype=np.dtype(obj[b'type'])).reshape(obj[b'shape'])
            else:
                return np.fromstring(obj[b'data'],
                            dtype=np.dtype(obj[b'type']))[0]
        elif 'complex' in obj:
            return complex(obj[b'data'])
        else:
            return obj
    except KeyError:
        return obj

def save_msgpack(args):
    print("Loading embeddings.")
    fname = "{}/labels.csv".format(args.feature_prefix)
    labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
    labels = map(itemgetter(1),
                 map(os.path.split,
                     map(os.path.dirname, labels)))  # Get the directory.
    fname = "{}/reps.csv".format(args.feature_prefix)
    embeddings = pd.read_csv(fname, header=None).as_matrix()
    #print type(labels)
    #print labels
    #print type(embeddings)
    #print embeddings
    # this is finding one entry for each name, and it always overwrite
    # if the name is the same, therefore always the last rep is kept
    dictionary = dict(zip(labels, embeddings))
    # print dictionary
    fmp = "{}/feature_db.mp".format(args.feature_prefix)
    with open(fmp, 'w') as fmpout:
		mp.pack(dictionary, fmpout, default=encode)
    #with open(fmp, 'r') as fmpin:
    #    unpacked = mp.unpack(fmpin, object_hook=decode)
    #    print unpacked

def infer(args, imgname):
    with open(args.classifier_model, 'r') as f:
        (le, clf) = pickle.load(f)

    rep = get_rep(args, readimg(imgname, args.size))
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

def detect_frame(args, framegray, align):
    start = time.time()
    # Get the largest face bounding box
    # bb = align.getLargestFaceBoundingBox(framegray) #Bounding box

    # Get all bounding boxes
    bb = align.getAllFaceBoundingBoxes(framegray)
    print("  Face detection took {} seconds.".format(time.time() - start))
    if bb is None:
        # raise Exception("Unable to find a face: {}".format(imgPath))
        return None
    print("  Face detection got {} faces.".format(len(bb)))
    return bb

def align_frame(args, framegray, align, bb):
    start = time.time()
    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.size,
                framegray,
                box,
                landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE))
    print("  Alignment took {} seconds.".format(time.time() - start))
    if alignedFaces is None:
        raise Exception("Unable to align the frame")
        return None
    print("  Face align {} faces.".format(len(alignedFaces)))
    return alignedFaces

def get_reps_frame(args, alignedFaces):
    start = time.time()
    reps = []
    for alignedFace in alignedFaces:
        # cv2.imshow('face', alignedFace)
        img = np.expand_dims(alignedFace, axis=0)
        arr = np.zeros((1, 1, args.size, args.size), dtype=float)
        arr[0][:] = img/255.0
        #reps.append(net.forward(alignedFace))
        reps.append(get_rep(args, arr))
    print("  Neural network forward pass took {} seconds.".format(time.time() - start))
    return reps

def infer_frame(args, framegray, align):
    with open(args.classifier_model, 'r') as f:
        (le, clf) = pickle.load(f)

    bb = detect_frame(args, framegray, align)
    if bb == None:
        return None
    alignedFaces = align_frame(args, framegray, align, bb)
    if alignedFaces == None:
        return None
    reps = get_reps_frame(args, alignedFaces)
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print "No Face detected"
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print predictions
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print str(le.inverse_transform(max2)) + ": "+str( predictions [max2])
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        #if args.verbose:
        print("  Prediction took {} seconds.".format(time.time() - start))
            #pass
        # print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences)

def infer_camera(args):
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    video_capture = cv2.VideoCapture(0)
    #video_capture.set(3, 320)
    #video_capture.set(4, 240)
    align = AlignDlib(args.dlib_face_predictor)

    confidenceList = []
    while True:
        start = time.time()
        ret, frame = video_capture.read()
        framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("Capture took {} seconds.".format(time.time() - start))
        print("  Frame gray shape {}".format(framegray.shape))

        start = time.time()
        persons, confidences = infer_frame(args, framegray, align)
        print("  Infer took {} seconds.".format(time.time() - start))
        print "P: " + str(persons) + " C: " + str(confidences)
        try:
            # append with two floating point precision
            confidenceList.append('%.2f' % confidences[0])
        except:
            # If there is no face detected, confidences matrix will be empty.
            # We can simply ignore it.
            pass

        for i, c in enumerate(confidences):
            if c <= args.threshold:  # 0.5 is kept as threshold for known face.
                persons[i] = "_unknown"
        # Print the person name and conf value on the frame
        cv2.putText(frame, "P: {} C: {}".format(persons, confidences),
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('', frame)
        # quit the program on the press of key 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

# tp align
# ./align-dlib.py ./raw align outerEyesAndNose ./aligned --size 128
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--singlerep', type=int, default=0,
                        help='Do single image rep')
    parser.add_argument('--time', type=int, default=0,
                        help='Do time measure on single image rep')
    parser.add_argument('--feature', type=int, default=0,
                        help='Do feature extraction')
    parser.add_argument('--msgpack', type=int, default=0,
                        help='Save the reps and labels to a msgpack file')
    parser.add_argument('--train', type=int, default=0,
                        help='Do train')
    parser.add_argument('--infer', type=int, default=0,
                        help='Do infer')
    parser.add_argument('--camera', type=int, default=0,
                        help='Do infer with camera input')
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
    parser.add_argument('--dlib-face-predictor', default='../model/dlib/shape_predictor_68_face_landmarks.dat',
                        help='The dlib face predictor')
    parser.add_argument('--ldaDim', type=int, default=-1)
    parser.add_argument('--threshold', type=float, default=0.5)
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
        get_rep(args, readimg(args.singleimg, args.size))
        sys.exit(0)
    if args.time == 1:
        if not os.path.isfile(args.singleimg):
            logging.info("Test Image not present.")
            sys.exit(-1)
        get_rep(args, readimg(args.singleimg, args.size), True, 20)
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
    if args.msgpack == 1:
        if not os.path.exists(args.feature_prefix):
            logging.info("Error: feature-prefix not present.")
            sys.exit(-1)
        save_msgpack(args)
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
    if args.camera == 1:
        if not os.path.isfile(args.classifier_model):
            logging.info("Error: classifier-model not present.")
            sys.exit(-1)
        infer_camera(args)
        sys.exit(0)

if __name__ == '__main__':
    main()
