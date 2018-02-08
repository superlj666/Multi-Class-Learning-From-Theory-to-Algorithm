#!/usr/bin/env python
import sys
sys.path.append('/home/bd-dev/lijian/201801_ICML/script/libsvm/libsvm-3.22/python')
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
mem = Memory("./mycache")

import scipy.io as sio
import scipy
import numpy as np
from svmutil import *
from svm import *
from shogun import LibSVM, KernelMulticlassMachine, MulticlassOneVsRestStrategy
from shogun import CombinedFeatures, RealFeatures, MulticlassLabels
from shogun import CombinedKernel, GaussianKernel, LinearKernel,PolyKernel
from shogun import MKLMulticlass
from shogun import GMNPSVM, CSVFile
from sklearn.model_selection import train_test_split
from shogun import MulticlassLibLinear
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from tempfile import NamedTemporaryFile

def feature_extract(X):
    pca = PCA(n_components=20)
    new_data=pca.fit_transform(X.toarray())
    return new_data

# load data from txt
def loadFromTxt (file_path):
    return np.loadtxt(file_path)

# load data from libsvm data
def loadFromLibsvm (data_name):
    data_path = '/home/bd-dev/lijian/201801_ICML/tmp/'+ data_name +'.scale'
    data = load_svmlight_file(data_path)
    X, y = data[0], data[1].reshape(len(data[1]),1)
    y=y-y.min()
    if data_name=='glass' or data_name=='svmguide4':    
        y = np.array([(x-1 if x>3 else x) for x in y])
    # if data_name=='sector':
    #     X=X[(y<10).reshape(6412,),:]
    #     X=feature_extract(X)
    #     y=y[(y<10).reshape(6412,),:]
    return X,y

# load data from .mat file
def loadFromMat (data_name):
    data_path = '/home/bd-dev/lijian/201801_ICML/tmp/RawData/'+ data_name +'/'+ data_name +'.phylpro.mat'
    label_path = '/home/bd-dev/lijian/201801_ICML/data/' + data_name + '/label_' + data_name + '.mat'
    data = sio.loadmat(data_path)['phylpros']
    label = sio.loadmat(label_path)['y']
    label=label-label.min()
    return data, label

# OneVsRest learning machine
def classifier_multiclassmachine (fm_train_real,fm_test_real,label_train_multiclass,width, C, epsilon):
    feats_train=RealFeatures(fm_train_real)
    feats_test=RealFeatures(fm_test_real)
    print '2 in'
    kernel=GaussianKernel(feats_train, feats_train, width)
    print '2 out'
    
    labels=MulticlassLabels(label_train_multiclass)

    classifier = LibSVM()
    classifier.set_epsilon(epsilon)
    mc_classifier = KernelMulticlassMachine(MulticlassOneVsRestStrategy(),kernel,classifier,labels)
    mc_classifier.train()

    kernel.init(feats_train, feats_test)
    out = mc_classifier.apply().get_labels()
    return out

# mc-mkl learning machine
def mkl_multiclass (fm_train_real, fm_test_real, label_train_multiclass,
    C, epsilon, num_threads, mkl_epsilon, mkl_norm):
    kernel = CombinedKernel()
    feats_train = CombinedFeatures()
    feats_test = CombinedFeatures()

    for i in range(-10,11):
        subkfeats_train = RealFeatures(fm_train_real)
        subkfeats_test = RealFeatures(fm_test_real)
        subkernel = GaussianKernel(pow(2,i+1))
        feats_train.append_feature_obj(subkfeats_train)
        feats_test.append_feature_obj(subkfeats_test)
        kernel.append_kernel(subkernel)
        
    kernel.init(feats_train, feats_train)

    labels = MulticlassLabels(label_train_multiclass)

    mkl = MKLMulticlass(C, kernel, labels)

    mkl.set_epsilon(epsilon)
    mkl.parallel.set_num_threads(num_threads)
    mkl.set_mkl_epsilon(mkl_epsilon)
    mkl.set_mkl_norm(mkl_norm)

    mkl.train()

    kernel.init(feats_train, feats_test)

    out =  mkl.apply().get_labels()
    return out

# multi-class classification based on C&S formulation
def classifier_multiclassliblinear (fm_train_real,fm_test_real,label_train_multiclass,width, C, epsilon):
    feats_train=RealFeatures(fm_train_real)
    feats_test=RealFeatures(fm_test_real)

    labels=MulticlassLabels(label_train_multiclass)

    classifier = MulticlassLibLinear(C,feats_train,labels)
    classifier.parallel.set_num_threads(num_threads)
    classifier.train()

    label_pred = classifier.apply(feats_test)
    out = label_pred.get_labels()
    return out

# multi-class on gmnp
def classifier_gmnpsvm (fm_train_real,fm_test_real,label_train_multiclass,width, C, epsilon):
    feats_train=RealFeatures(fm_train_real)
    feats_test=RealFeatures(fm_test_real)
    kernel= GaussianKernel(feats_train, feats_train, width)
    import time
    start=time.time()
    tmp=kernel.get_kernel_matrix();
    end=time.time()
    print 'use time: ' + str(end-start)

    labels=MulticlassLabels(label_train_multiclass)

    svm=GMNPSVM(C, kernel, labels)
    svm.set_epsilon(epsilon)
    svm.parallel.set_num_threads(num_threads)
    svm.train(feats_train)

    out=svm.apply(feats_test).get_labels()
    return out

def train_test(mode, X, y, C):
    accuracy=[]
    for i in range(times):        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)        
        if type(X_train)==scipy.sparse.csr.csr_matrix and type(X_test)==scipy.sparse.csr.csr_matrix:
            X_train=X_train.todense()
            X_test=X_test.todense()
        
        X_train=X_train.T
        X_test=X_test.T
        y_train=y_train.reshape(y_train.size,).astype('float64')
        y_test=y_test.reshape(y_test.size,).astype('float64')

        if mode=='mcmkl':
            label_pre = mkl_multiclass(X_train, X_test, y_train, C, epsilon, num_threads, mkl_epsilon, mkl_norm)
        elif mode=='1vR':
            label_pre = classifier_multiclassmachine(X_train, X_test, y_train, width, C, epsilon)
        elif mode=='gmnp':
            label_pre = classifier_gmnpsvm(X_train, X_test, y_train, width, C, epsilon)
        elif mode=='cs':
            label_pre = classifier_multiclassliblinear(X_train, X_test, y_train, width, C, epsilon)    
        accuracy.append((y_test==label_pre).sum()/float(label_pre.size))
        
        print 'finish '+ data_name + ' in ' + mode + ', round ' + str(i) +', accuracy: ' + str(accuracy[len(accuracy)-1])
    print 'mean accuracy of ' + data_name + ' in ' + mode + ' is ' + str(np.mean(accuracy))
    return accuracy

def cv_para(mode, X, y, C):
    accuracy=[]
    for train_index, test_index in KFold(n_splits=folds, shuffle=True).split(y):        
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        if type(X_train)==scipy.sparse.csr.csr_matrix and type(X_test)==scipy.sparse.csr.csr_matrix:
            X_train=X_train.todense()
            X_test=X_test.todense()
        
        X_train=X_train.T
        X_test=X_test.T
        y_train=y_train.reshape(y_train.size,).astype('float64')
        y_test=y_test.reshape(y_test.size,).astype('float64')

        if mode=='mcmkl':
            label_pre = mkl_multiclass(X_train, X_test, y_train, C, epsilon, num_threads, mkl_epsilon, mkl_norm)
        elif mode=='1vR':
            label_pre = classifier_multiclassmachine(X_train, X_test, y_train, width, C, epsilon)
        elif mode=='gmnp':
            label_pre = classifier_gmnpsvm(X_train, X_test, y_train, width, C, epsilon)
        elif mode=='cs':
            label_pre = classifier_multiclassliblinear(X_train, X_test, y_train, width, C, epsilon)    
        accuracy.append((y_test==label_pre).sum()/float(label_pre.size))
    print 'C: ' + str(C) + ' and mean accuracy of ' + data_name + ' in ' + mode + ' is ' + str(np.mean(accuracy))
    return np.mean(accuracy)

def get_best_para():
    best_para = 0
    max_acc = 0
    for para in para_list:
        C = para
        if file_type =='4':
            data, label = loadFromMat(data_name)
            accuracy =cv_para(mode, data, label, C)
        elif file_type =='5':
            X, y = loadFromLibsvm(data_name)
            accuracy = cv_para(mode, X, y, C)
        if max_acc < accuracy:
            max_acc = accuracy
            best_para = C
    return best_para


def combined_kernel(file_type, data_name, operate_type):
    if file_type == '4':
        X, y = loadFromMat(data_name)
    elif file_type == '5':
        X, y = loadFromLibsvm(data_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    if type(X_train) == scipy.sparse.csr.csr_matrix and type(X_test) == scipy.sparse.csr.csr_matrix:
        X_train = X_train.todense()
        X_test = X_test.todense()
    X_train = X_train.T
    X_test = X_test.T
    y_train = y_train.reshape(y_train.size, ).astype('float64')
    y_test = y_test.reshape(y_test.size, ).astype('float64')

    kernel = CombinedKernel()
    feats_train = CombinedFeatures()
    feats_test = CombinedFeatures()
    subkfeats_train = RealFeatures(X_train)
    subkfeats_test = RealFeatures(X_test)
    for i in range(-10, 11):
        subkernel = GaussianKernel(pow(2, i + 1))
        feats_train.append_feature_obj(subkfeats_train)
        feats_test.append_feature_obj(subkfeats_test)
        kernel.append_kernel(subkernel)
    kernel.init(feats_train, feats_train)
    tmp_train_csv = NamedTemporaryFile(suffix=data_name + '_combined.csv')

    import time
    start = time.time()
    if operate_type == 'save':
        km_train = kernel.get_kernel_matrix()
        f = CSVFile(tmp_train_csv.name, "w")
        kernel.save(f)
    elif operate_type == 'load':
        f = CSVFile(tmp_train_csv.name, "r")
        kernel.load(f)
    end = time.time()
    print 'for saving or loading, use time : ' + str(end - start)

    labels = MulticlassLabels(y_train)

    mkl = MKLMulticlass(C, kernel, labels)

    mkl.set_epsilon(epsilon)
    mkl.parallel.set_num_threads(num_threads)
    mkl.set_mkl_epsilon(mkl_epsilon)
    mkl.set_mkl_norm(mkl_norm)

    import time
    start = time.time()
    mkl.train()
    end = time.time()
    print 'use time : ' + str(end - start)

    kernel.init(feats_train, feats_test)
    out = mkl.apply().get_labels()
    print out.shape
    print sum(out == y_test) / float(len(out))

# times=30
# C=100 #[1, 10, 100, 1000]
# epsilon=1e-1
# mkl_epsilon=0.001
# test_size=0.2
# width=8
# num_threads=32
# mkl_norm=1
#
# combined_kernel('5', 'dna','load')



if __name__ == '__main__':
    times = 30
    C = 100  # [1, 10, 100, 1000]
    epsilon = 0.1 #1e-5
    mkl_epsilon = 0.1#0.001
    test_size = 0.2
    width = pow(2, -8)
    num_threads = 32
    mkl_norm = 1

    #sys.argv=['1','5','sector','gmnp','1']
    file_type = sys.argv[1]
    data_name = sys.argv[2]
    mode = sys.argv[3]

    if len(sys.argv) > 4:
        C = float(sys.argv[4])
    if len(sys.argv) > 5:
        folds = int(sys.argv[5])
        para_list = np.logspace(-2, 12, 15, base=2)
        C = get_best_para()
        print 'best para is ' + str(C)

    if file_type == '4':
        data, label = loadFromMat(data_name)
        accuracy = train_test(mode, data, label, C)
    elif file_type == '5':
        X, y = loadFromLibsvm(data_name)
        accuracy = train_test(mode, X, y, C)
    print("\n".join(str(item * 100) for item in accuracy))

