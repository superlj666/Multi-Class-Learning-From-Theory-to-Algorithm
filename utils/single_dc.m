function performance = single_dc(Ktrain, Ytrain, Ktest, Ytest, localR, C, gamma)
% Parameters for UFO-MKL
model_zero         = model_init();
model_zero.n_cla   = max(max(Ytest),max(Ytrain));
model_zero.T       = 300;   % Number of epochs
model_zero.lambda  = 1/(C*numel(Ytrain));%1/(C*numel(Ytrain));
model_zero.gamma   = gamma;
model_zero.localR  = localR;


model_zero.step   = 10*numel(Ytrain);
options.eachRound = @ufomkl_test;
options.Ktest     = Ktest;
options.Ytest     = Ytest;

model = new_k_ufomkl_dc_train(Ktrain, Ytrain, model_zero, options);
performance=max(model.acc1);