addpath( './utils' );
clear
close all

datasets={'usps'};
size_arr=[7291];
data_path=['data/usps/Gaussian_-3.mat'];
best_para=[8]

all_results=zeros(30, length(datasets));
for j=1:length(datasets)
%     [kernel_name, best_para]=choose_kernel_1vRest(char(datasets(j)))
%     data_path=kernel_name;
    
    label_path=['data/labels/label_', char(datasets(j)),'.mat'];
    load(data_path);
    load(label_path);
    if strcmp(char(datasets(j)),'glass') || strcmp(char(datasets(j)),'svmguide4')
        label_vector(label_vector>3) = label_vector(label_vector>3)-1;
    end
    sample_n=size_arr(j);
    train_part=0.8;
    
    rand('state', 0);
    for i=1:30
        rand_arr = randperm(sample_n);
        
        train_array=rand_arr(1:sample_n*train_part);
        test_array=rand_arr(sample_n*train_part+1:sample_n);
        test_size=size(test_array,2);
        train_size=size(train_array,2);
        
        training_label_vector=label_vector(train_array)';
        training_instance_matrix=[(1:train_size)',K(train_array,train_array)];
        testing_label_vector=label_vector(test_array)';
        testing_instance_matrix=[(1:test_size)',K(test_array,train_array)];
        
        numLabels=max(training_label_vector);
        para=strcat('-c', 32, num2str(best_para), ' -t 4 -b 1 -q');
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
        all_results(i,j)=acc*100;
    end
end
