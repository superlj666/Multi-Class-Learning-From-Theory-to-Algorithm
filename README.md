# ICML_2018_Local_Rademancher
Paper and experiments in ICML_2018_Local_Rademancher, own by Yong Liu and Jian Li

## Usage of source code in Multi-class Kernel Learning: Fast Rate and Algorithms
#### Enviroment
We do experiments based on following softwares:
1. Python 2.7
2. MATLAB R2017b
3. DOGMA toolbox from http://dogma.sourceforge.net/
4. SHOGUN-6.1.3 from https://github.com/shogun-toolbox/shogun
5. LIBSVM Tools from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/
6. sklearn for python
#### Data sets
1. plant, psortPos, psortNeg and nonpl from http://www.raetschlab.org/suppl/protsubloc
2. others from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
#### Steps
1. Download data sets and move dataName.phylpro.mat, label_dataName.mat and dataName.scale to ./data/
2. Create Gaussian kernels: change variable file_list in Test_Gaussian_Kernel.m and run
3. SMSD-MKL: change variables of data_sets in Test_SMSD_MKL.m and run
4. Conv-MKL: change variables of data_sets in Test_Conv_MKL.m and run
5. LMC: change variables of data_sets in Test_LMC.py and run
6. OneVsOne: change variables of data_sets in Test_OneVsOne.m and run
7. OneVsRest: change variables of data_sets in Test_OneVsOne.m and run
7. GMNP: change variables of data_sets in Test_OneVsOne.m and run
7. l1 MC-MKL: change variables of data_sets in Test_OneVsOne.m and run
7. l2 MC-MKL: change variables of data_sets in Test_OneVsOne.m and run
7. UFO-MKL: change variables of data_sets in Test_UFO_MKL.m and run



uuu 
2018/1/29 Add code of experiments 

2018/1/30 Add statistic files of experiments
