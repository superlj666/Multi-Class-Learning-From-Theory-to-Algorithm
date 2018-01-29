clear;
close all;
train_part = 0.8;
rounds=30;
folds=5;
% datasets={'psortPos', 'vehicle','glass','dna', 'svmguide2'};
% size_arr=[541,846,214,2000,391];

datasets={'psortPos'};
size_arr=[541];
C_list=2.^(-2:1:12);

all_results=zeros(rounds, length(datasets));
for j=1:length(datasets)
    sample_n=size_arr(j);
    [KMatrix, label_vector] = load_data(char(datasets(j)), 'mkl');
    
    best_para = get_best_para(KMatrix, label_vector, {C_list}, 'mkl', folds);
    fprintf('best C is %.2f\n',best_para(end));
    
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
    fprintf('mean acc of %s is %.2f and best C is %.2f\n ',char(datasets(j)), mean(all_results(:,j)), best_para(end));
end