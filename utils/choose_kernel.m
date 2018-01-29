function   [kernel_name, para]=choose_kernel(data_name)

c_list=2.^(-2:1:12);
folds=30;
str1=['/home/bd-dev/lijian/201801_ICML/data/new_kernels/',data_name,'/Gaussian_'];
label_path=['/home/bd-dev/lijian/201801_ICML/data/labels/label_', data_name,'.mat'];
load(label_path);
sample_n = size(label_vector,2);
str2=strsplit(num2str(-10:1:10),' ');
str3='.mat';
str_arr=strcat(str1, char(str2), str3);

performance_kernel=[];
para_candidate=[];
for index=1:size(str_arr,1)
    load(strtrim(str_arr(index,:)));
    
    performance_cv=zeros(length(c_list),1);
    for index=1:length(c_list)
        rand('state', 0);
        performance_c=zeros(folds,1);
        indices = crossvalind('Kfold',label_vector,folds);
        for i = 1:folds
            test_array = (indices == i); train_array = ~test_array;
            
            training_label_vector=label_vector(train_array)';
            training_instance_matrix=[(1:sum(train_array))',K(train_array,train_array)];
            testing_label_vector=label_vector(test_array)';
            testing_instance_matrix=[(1:sum(test_array))',K(test_array,train_array)];
            
            para=strcat('-c', 32, num2str(c_list(index)), ' -t 4 -q');
            model = svmtrain(training_label_vector, training_instance_matrix, para);
            [predicted_label, accuracy, prob_estimates] = svmpredict(testing_label_vector, testing_instance_matrix, model);
            
            performance_c(i)=accuracy(1);
        end
        performance_cv(index)=mean(performance_c);
    end
    [value, loc]=max(performance_c);
    performance_kernel(end+1)=value;
    para_candidate(end+1)=c_list(loc);
end

[max_acc, kernel_loc]=max(performance_kernel);
kernel_name=strtrim(str_arr(kernel_loc,:));
para=para_candidate(loc);
