%% A test script for MVLLE clustering
clear
clc
addpath('./libs/')
%% init
main_opts.K = [45];

%% load dataste
main_opts.dataset = '3Sources_169_3views_6clusters';
load(strcat('./datasets/', main_opts.dataset), 'X', 'truth')
V = numel(X);
N = size(X{1}, 2);
clusters = length(unique(truth));

%% MVLLE parameters
multiViewLLE_settings.eta = 0.1; % EG learning rate
multiViewLLE_settings.gamma = 1; % smooth weight (not required for SWMVLLE)
multiViewLLE_settings.display = true;
multiViewLLE_settings.lle_max_iter = 200;

%% main loop
ACC = zeros(length(main_opts.K), 1);
for idx=1:length(main_opts.K)
    multiViewLLE_settings.K = main_opts.K(idx);
    if multiViewLLE_settings.K * V >= N
        break
    end
%     W = multiViewLLE(X, multiViewLLE_settings);
    W = SWMVLLE(X, multiViewLLE_settings);
    %% clustering
    Z = 0.5*(abs(W)+abs(W'));
    grps = SpectralClustering(Z, clusters);
    
    %% evaluation
    P_label = bestMap(truth, grps);
    ACC(idx) = length(find(truth == P_label))/length(truth);
    fprintf('K=%d, acc=%0.4f\n', multiViewLLE_settings.K, ACC(idx))
end