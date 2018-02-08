function [KMatrix, label_vector, localR] = load_data(data_name, type)
str1=['data/',data_name,'/Gaussian_'];
label_path=['data/labels/label_', data_name,'.mat'];
load(label_path);
sample_n = size(label_vector,2);
str2=strsplit(num2str(-10:1:10),' ');
str3='.mat';
str_arr=strcat(str1, char(str2), str3);
kernel_size=length(str2);
KMatrix=zeros(sample_n,sample_n,kernel_size);

tail_size = sample_n-2;
localR=zeros(1, kernel_size);
for index=1:size(str_arr,1)
    load(strtrim(str_arr(index,:)));
    
    % Using Local Rademacherqon
    e=eig(K);
    
    KMatrix(:,:,index)=single(K);
    localR(index)=sum(e(1:tail_size));
    if strcmp(type, 'conv')
        KMatrix(:,:,index)=single(K*sample_n/localR(index));
    end
end