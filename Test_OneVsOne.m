addpath( './utils' );
clear
close all

% datasets={'segment','vehicle','vowel','wine','dna','glass','iris', 'svmguide2','svmguide4','satimage'};
% size_arr=[2310,846,528,178,2000,214,150,391,300,4435];

datasets={'usps'};
size_arr=[7291];
all_results=zeros(30, length(datasets));
for j=1:length(datasets)
    [kernel_name, best_para]=choose_kernel(char(datasets(j)))
    data_path=kernel_name;
    
    label_path=['data/labels/label_', char(datasets(j)),'.mat'];
    load(data_path);
    load(label_path);
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
        
        para=strcat('-c', 32, num2str(best_para), ' -t 4 -q');
        model = svmtrain(training_label_vector, training_instance_matrix, para);
        [predicted_label, accuracy, prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model);
        
        all_results(i,j)=accuracy(1);
    end
end
