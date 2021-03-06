addpath( './utils' );
addpath('../libsvm/libsvm-3.22/matlab/')
clear
close all
% datasets={'plant','psortPos', 'psortNeg', 'nonpl', 'sector', 'segment','vehicle','vowel','wine','dna','glass','iris', 'svmguide2','satimage', 'usps'};

datasets={'iris'};
C_list=2.^(-2:1:12);

train_part = 0.8;
rounds=50;
folds=10;
all_results=zeros(rounds, length(datasets));
for j=1:length(datasets)
    [KMatrix, label_vector] = load_kernels(char(datasets(j)), 'conv');
    sample_n=length(label_vector);
    
    best_para = get_best_para(KMatrix, label_vector, {C_list}, 'conv', folds);
    fprintf('best C is %.6f\n',best_para(end));
    
    rand('state', 0);
    for i=1:rounds
        rand_arr = randperm(sample_n);
        
        train_array=rand_arr(1:int64(sample_n*train_part));
        test_array=rand_arr(int64(sample_n*train_part)+1:sample_n);
        
        Ktrain=KMatrix(train_array,train_array,:);
        Ktest=KMatrix(train_array,test_array,:);
        Ytrain=label_vector(train_array);
        Ytest=label_vector(test_array);
        
        all_results(i, j)=single_mkl(Ktrain, Ytrain, Ktest, Ytest, best_para(end));
        fprintf('for %s round:%2d acc:%.2f\n',char(datasets(j)), i, all_results(i, j));
    end
    fprintf('mean acc of %s is %.2f and best C is %.6f\n ',char(datasets(j)), mean(all_results(:,j)), best_para(end));
end
