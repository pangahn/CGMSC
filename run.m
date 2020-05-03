clear
clc
addpath('./libs/')

%% CMSMSC
main_opts.save_result = false; % Whether to save the results to local file
main_opts.normalize_method = 'L2';
main_opts.sub_fix = 'final'; % Suffix of result file name

%% MISC
opts.display = true;

%% MVLLE
multiViewLLE_settings.lle_file = true; % whether to use local weight file
multiViewLLE_settings.eta = 0.1; % EG learning rate (remain unchanged)
multiViewLLE_settings.gamma = 1; % smooth weight (not required for SWMVLLE)

%%
main_opts.dataset = '3Sources_169_3views_6clusters';
% main_opts.dataset = 'BBCsport_544_2views_5clusters';
% main_opts.dataset = 'NGs_500_3views_5clusters';

main_opts.M = 1;
main_opts.K = [20];
lambda_list.lambda1 = [10];
lambda_list.lambda2 = [1];
lambda_list.lambda3 = [0];
main(main_opts, lambda_list, multiViewLLE_settings, opts)
