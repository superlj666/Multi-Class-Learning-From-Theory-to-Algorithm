% create Guassian Kernel Matrices where \tau in [-20,20]
data_name='usps';
[label_vector, instance_matrix] = libsvmread(['data/',data_name, '.scale']);
tau = -10:1:10;
sample_n = size(instance_matrix,1);
for t=1:length(tau)    
    norms = sum(instance_matrix'.^2);
    K = exp((-norms'*ones(1,sample_n) - ones(sample_n,1)*norms + 2*(instance_matrix*instance_matrix'))/(2*2^tau(t)));
    
    str=['data/', data_name,'/Gaussian_',num2str(tau(t)),'.mat'];
    save(str, 'K');
end

label_vector=label_vector+1-min(label_vector);
if size(label_vector,1) ~= 1
    label_vector=label_vector';
end
save(['data/labels/label_',data_name,'.mat'],'label_vector');
