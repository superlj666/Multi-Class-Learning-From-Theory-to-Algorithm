addpath( './utils' );
clear
close all
% file_list={'plant','psortPos', 'psortNeg', 'nonpl', 'sector', 'segment','vehicle','vowel','wine','dna','glass','iris', 'svmguide2','satimage', 'usps'};
file_list={'iris'};

for i=1:length(file_list)    
    data_name=char(file_list(i));
    if strcmp(data_name,'plant') || strcmp(data_name,'psortPos') || strcmp(data_name,'psortNeg') || strcmp(data_name,'nonpl')
        load(['data/', data_name, '.phylpro.mat']);
        load(['data/label_', data_name, '.mat']);
        instance_matrix=phylpros;
        label_vector=y;
    else
        [label_vector, instance_matrix] = libsvmread(['data/',data_name, '.scale']);
    end
    gaussian_kernel(data_name, label_vector, instance_matrix);
end