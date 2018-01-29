clear
% 1. valid those two .mat files are corresponding.
% load('/home/bd-dev/lijian/201801_ICML/tmp/RawData/plant/plant.phylpro.mat');
% load('/home/bd-dev/lijian/201801_ICML/data/plant/label_plant.mat'); 
% instance_matrix=phylpros;
% indices = crossvalind('Kfold',y,10);
% cp = classperf(y);
% for i = 1:10
%     test = (indices == i); train = ~test;
% 	class = classify(phylpros(test,:),phylpros(train,:),y(train,:));
% 	classperf(cp,class,test);
% end
% cp.ErrorRate

% 2. create Guassian Kernel Matrices where \tau in [-20,20]
% [label_vector1, instance_matrix1] = libsvmread('/home/bd-dev/lijian/icml_2018/tmp/rcv1_test.multiclass');
% [label_vector2, instance_matrix2] = libsvmread('/home/bd-dev/lijian/icml_2018/tmp/rcv1_train.multiclass');
% label_vector=[label_vector1;label_vector2];
% instance_matrix=[instance_matrix1;instance_matrix2];
% tau = -20:1:20;
% sample_n = size(instance_matrix,1);
% for t=1:length(tau)    
%     norms = sum(instance_matrix'.^2);
%     K = exp((-norms'*ones(1,sample_n) - ones(sample_n,1)*norms + 2*(instance_matrix*instance_matrix'))/(2*2^tau(t)));
%     
%     str=['/home/bd-dev/lijian/icml_2018/new_kernels/rcv1/Gaussian_',num2str(tau(t)),'.mat'];
%     save(str, 'K');
% end

% 3. save labels named Y which starts with 1 and scale to 1*n_cla
% load('/home/bd-dev/lijian/201801_ICML/data/nonpl/label_nonpl.mat');
% label_vector=y'+1;
% save('/home/bd-dev/lijian/201801_ICML/data/labels/label_nonpl.mat','label_vector');


% 4. from sparse matrix to full matrix