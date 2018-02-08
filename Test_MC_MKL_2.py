#!/usr/bin/env python
import numpy as np
import sys
from class_tools import *

if __name__ == '__main__':
    #datasets=['plant','psortPos', 'psortNeg', 'nonpl', 'sector', 'segment','vehicle','vowel','wine','dna','glass','iris', 'svmguide2','satimage', 'usps']
    
    datasets=['iris']
    C_list = np.logspace(-2, 12, 15, base=2)

    times = 50
    epsilon = 1e-5
    test_size = 0.2
    folds = 10

    for i in range(len(datasets)):
        if datasets[i] in ['plant','psortPos', 'psortNeg', 'nonpl']:
            file_type='4'
        else:
            file_type='5'
        data_name = datasets[i]
        mode='mcmkl1'
        para_list=[C_list, data_name, mode, file_type]
        C = get_best_para(para_list)

    if file_type == '4':
        data, label = loadFromMat(data_name)
        accuracy = train_test(mode, data, label, C, data_name)
    elif file_type == '5':
        X, y = loadFromLibsvm(data_name)
        accuracy = train_test(mode, X, y, C, data_name)
    print("\n".join(str(item * 100) for item in accuracy))

