function   [kernel_name, para]=choose_kernel_1vRest(data_name, c_list, folds)

str1=['data/',data_name,'/Gaussian_'];
label_path=['data/labels/label_', data_name,'.mat'];
load(label_path);
if strcmp(data_name,'glass') || strcmp(data_name,'svmguide4')
    label_vector(label_vector>3) = label_vector(label_vector>3)-1;
end

sample_n = size(label_vector,2);
str2=strsplit(num2str(-10:1:10),' ');
str3='.mat';
str_arr=strcat(str1, char(str2), str3);

performance_kernel=[];
para_candidate=[];
for index=1:size(str_arr,1)
    load(strtrim(str_arr(index,:)));
    
    performance_cv=zeros(length(c_list),1);
    for j=1:length(c_list)
        rand('state', 0);
        performance_c=zeros(folds,1);
        indices = crossvalind('Kfold',label_vector,folds);
        for i = 1:folds
            test_array = (indices == i); train_array = ~test_array;
            
            training_label_vector=label_vector(train_array)';
            training_instance_matrix=[(1:sum(train_array))',K(train_array,train_array)];
            testing_label_vector=label_vector(test_array)';
            testing_instance_matrix=[(1:sum(test_array))',K(test_array,train_array)];
            
            numLabels=max(training_label_vector);
            para=strcat('-c', 32, num2str(c_list(j)), ' -t 4 -b 1 -q');
            model = cell(numLabels,1);
            for k=1:numLabels
                model{k} = svmtrain(double(training_label_vector==k), training_instance_matrix, para);
            end
            
            prob = zeros(sum(test_array),numLabels);
            for k=1:numLabels
                [~,~,p] = svmpredict(double(testing_label_vector==k), testing_instance_matrix, model{k}, '-b 1');
                prob(:,k) = p(:,model{k}.Label==1); %# probability of class==k
            end
            [~,pred] = max(prob,[],2);
            acc = sum(pred == testing_label_vector) ./ numel(testing_label_vector) %# accuracy
            performance_c(i)=acc*100;
        end
        performance_cv(j)=mean(performance_c);
    end
    [value, loc]=max(performance_cv);
    performance_kernel(end+1)=value;
    para_candidate(end+1)=c_list(loc);
end

[max_acc, kernel_loc]=max(performance_kernel);
kernel_name=strtrim(str_arr(kernel_loc,:));
para=para_candidate(kernel_loc);
