function best_para=get_best_para(KMatrix, label_vector, paralist, mode, folds)
best_para=[];

if strcmp(mode, 'dc')
    C_list = cell2mat(paralist(1));
    gamma_list = cell2mat(paralist(2));
    C_accuracy = [];
    for j=1:length(C_list)
        for k=1:length(gamma_list)
            rand('state', 0);
            cv_accuracy=[];
            C=C_list(j);
            indices = crossvalind('Kfold',label_vector,folds);
            for i = 1:folds
                test = (indices == i); train = ~test;
                Ktrain=KMatrix(train,train,:);
                Ktest=KMatrix(train,test,:);
                Ytrain=label_vector(train);
                Ytest=label_vector(test);
                cv_accuracy(end+1)=single_mkl(Ktrain, Ytrain, Ktest, Ytest, C);
            end
            C_accuracy(end+1)=mean(cv_accuracy);
        end
    end
    
    [value, loc]=max(C_accuracy);    
    
    best_para(end+1)=C_list(fix(loc/length(C_list))+1);
    best_para(end+1)=gamma_list(mod(loc-1, length(gamma_list))+1);
else
    C_list = cell2mat(paralist(1));
    C_accuracy = [];
    for j=1:length(C_list)
        rand('state', 0);
        cv_accuracy=[];
        C=C_list(j);
        indices = crossvalind('Kfold',label_vector,folds);
        for i = 1:folds
            test = (indices == i); train = ~test;
            Ktrain=KMatrix(train,train,:);
            Ktest=KMatrix(train,test,:);
            Ytrain=label_vector(train);
            Ytest=label_vector(test);
            cv_accuracy(end+1)=single_mkl(Ktrain, Ytrain, Ktest, Ytest, C);
        end
        C_accuracy(end+1)=mean(cv_accuracy);
    end
    
    [value, loc]=max(C_accuracy);
    best_para(end+1)=C_list(loc);
end