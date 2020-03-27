clear
clc
addpath('./libs/')

%% main.m 参数
main_opts.dataset = '3Sources_169_3views_6clusters';
% main_opts.dataset = 'BBCsport_544_2views_5clusters';
% main_opts.dataset = 'MSRC-v1_210_5views_7clusters';
% main_opts.dataset = 'UCI-digit_2000_6views_10clusters';
% main_opts.dataset = 'ORL_400_4views_40clusters';
% main_opts.dataset = 'COIL-20_1440_3views_20clusters';

main_opts.save_result = false;
main_opts.normalize_method = 'L2';
main_opts.sub_fix = 'final';
%% CMSMSC.m 参数
% 3Sources:   10,   90,     12.5893,  20,,,,56]
% BBCsport:    5,   30,     79.4328,  20,,,,272]
% MSRCv1:     10,   0.1,    7.9433,   5,,,,40]
% ORL:        80,   0.001,  1.9953,   30,,,,100)
% UCI digits:  5,   70,     79.4328,  10,,,,333]
% COIL-20:     5,   100,    19.9526,  15,,,,480)

main_opts.M = 1;
main_opts.K = [10];
opts.display = false;

%% multiViewLLE.m 参数
multiViewLLE_settings.lle_file = true;
multiViewLLE_settings.eta = 0.1; % EG learning rate (no change)
multiViewLLE_settings.gamma = 1; % smooth weight
multiViewLLE_settings.display = false;
multiViewLLE_settings.lle_max_iter = 200;

%%
lambda_list.lambda1 = [10 30];
lambda_list.lambda2 = [30];
lambda_list.lambda3 = [79.4328];
main(main_opts, lambda_list, multiViewLLE_settings, opts)