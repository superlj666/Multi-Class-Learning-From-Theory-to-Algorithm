addpath( './utils' );
clear
close all

% datasets={'plant', 'psortPos', 'psortNeg', 'nonpl', 'sector', 'segment','vehicle','vowel','wine','dna','glass','iris', 'svmguide2','svmguide4','satimage'};
% size_arr=[940,541,1444,2732,6412,2310,846,528,178,2000,214,150,391,300,4435];
% datasets={'sector', 'segment','vehicle','vowel','wine','dna','glass','iris', 'svmguide2','svmguide4','satimage'};
% size_arr=[6412,2310,846,528,178,2000,214,150,391,300,4435];
datasets={'vehicle'};
size_arr=[846];
c_list= [1, 10,1000];%10.^(-2:5);
all_results=zeros(30, length(c_list), length(datasets));
for j=1:length(datasets)
    data_path=['/home/bd-dev/lijian/201801_ICML/data/new_kernels/',char(datasets(j)),'/Gaussian_4.mat'];
    label_path=['/home/bd-dev/lijian/201801_ICML/data/labels/label_', char(datasets(j)),'.mat'];
    load(data_path);
    load(label_path);
    sample_n=size_arr(j);
    train_part=0.8;
    
    performance_cv=zeros(length(c_list),1);
    for index=1:length(c_list)
        rand('state', 0);
        performance_c=zeros(30,1);
        for i=1:30
            rand_arr = randperm(sample_n);
            
            train_array=rand_arr(1:sample_n*train_part);
            test_array=rand_arr(sample_n*train_part+1:sample_n);
            test_size=size(test_array,2);
            train_size=size(train_array,2);
            
            if strcmp(char(datasets(j)),'glass') || strcmp(char(datasets(j)),'svmguide4')
                label_vector(label_vector>3) = label_vector(label_vector>3)-1;
            end
           
            training_label_vector=label_vector(train_array)';
            training_instance_matrix=[(1:train_size)',K(train_array,train_array)];
            testing_label_vector=label_vector(test_array)';
            testing_instance_matrix=[(1:test_size)',K(test_array,train_array)];
            
            numLabels=max(training_label_vector);
            para=strcat('-c', 32, num2str(c_list(index)), ' -t 4 -b 1 -q');
            model = cell(numLabels,1);
            for k=1:numLabels
                model{k} = svmtrain(double(training_label_vector==k), training_instance_matrix, para);
            end
             
            prob = zeros(test_size,numLabels);
            for k=1:numLabels
                [~,~,p] = svmpredict(double(testing_label_vector==k), testing_instance_matrix, model{k}, '-b 1');
                prob(:,k) = p(:,model{k}.Label==1); %# probability of class==k
            end
            [~,pred] = max(prob,[],2);
            acc = sum(pred == testing_label_vector) ./ numel(testing_label_vector) %# accuracy
            performance_c(i)=acc*100;
        end
        performance_cv(index)=mean(performance_c);
        all_results(:,index, j)=performance_c;
    end
end
