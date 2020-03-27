function main(main_opts, lambda_list, multiViewLLE_settings, opts)
%% 数据处理
load(strcat('./datasets/', main_opts.dataset), 'X', 'truth')
n_v = numel(X); %#ok<*NODEF>
if isfield(main_opts, 'normalize_method')
    for i = 1:n_v
        X{i}  = normalize(X{i}, main_opts.normalize_method); %#ok<*SAGROW>
    end
    if strcmp(main_opts.normalize_method, 'L2')
        main_opts.normalize_method = 1;
    elseif strcmp(main_opts.normalize_method, 'MinMax')
        main_opts.normalize_method = 2;
    elseif strcmp(main_opts.normalize_method, 'Inf')
        main_opts.normalize_method = 3;
    end
else
    main_opts.normalize_method = 0;
end

%% 结果保存
filename = fullfile('result', [main_opts.dataset ,'_' main_opts.sub_fix '.csv']);
if main_opts.save_result
    if ~exist(filename, 'file')
        header = {'Accuracy', 'NMI', 'Precision', 'Recall', 'F1', 'ARI',...
            'std_ACC', 'std_NMI', 'std_P', 'std_R', 'std_F1', 'std_ARI',...
            'lambda1', 'lambda2', 'lambda3', 'K', 'normalize', 'time'};
        fid = fopen(filename,'a+');
        fprintf(fid,'%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n',...
            header{1}, header{2}, header{3}, header{4}, header{5},...
            header{6}, header{7}, header{8}, header{9}, header{10},...
            header{11}, header{12}, header{13}, header{14}, header{15},...
            header{16}, header{17}, header{18});
        fclose(fid);
    end
end

%% 主循环
ACCi = zeros(main_opts.M, 1);
NMIi = zeros(main_opts.M, 1);
Pi = zeros(main_opts.M, 1);
Ri = zeros(main_opts.M, 1);
F1i = zeros(main_opts.M, 1);
ARIi = zeros(main_opts.M, 1);
time = zeros(main_opts.M, 1);
opts.lambda = zeros(3, 1);
clusters = length(unique(truth));
N = size(X{1}, 2);
for kkk=1:length(main_opts.K)
    if main_opts.K(kkk) * n_v >= N
        break
    end
    %% 计算一致流型
    multiViewLLE_settings.K = main_opts.K(kkk);
    lle_filename = strcat('./lle/', main_opts.dataset, '_',  ...
        num2str(multiViewLLE_settings.K), '.mat');
    % 如果本地已有数据则使用本地数据
    fprintf('running multiViewLLE.m K=%d ===\n', multiViewLLE_settings.K)
    if multiViewLLE_settings.lle_file && exist(lle_filename, 'file')
        load(lle_filename, 'W')
    end
    % 参数设置重新计算或本地不存在数据
    if ~multiViewLLE_settings.lle_file || ~exist(lle_filename, 'file')
        W = multiViewLLE(X, multiViewLLE_settings);
        save(strcat('./lle/', main_opts.dataset, '_',...
            num2str(multiViewLLE_settings.K), '.mat'), 'W')
    end
    for i=1:length(lambda_list.lambda1)
        opts.lambda(1) = lambda_list.lambda1(i);
        for j=1:length(lambda_list.lambda2)
            opts.lambda(2) = lambda_list.lambda2(j);
            for k=1:length(lambda_list.lambda3)
                opts.lambda(3) = lambda_list.lambda3(k);
                fprintf('lambda1=%f, lambda2=%f, lambda3=%f, K=%d\n', ...
                        opts.lambda(1), opts.lambda(2), opts.lambda(3), multiViewLLE_settings.K);
                for m=1:main_opts.M
                    fprintf('>>> %d of %d ', m, main_opts.M);

                    tic1 = tic;
                    try
                        [Z, alpha] = CMSMSC(X, W, opts);
                        Zstar = zeros(length(truth));
                        for v=1:n_v
                            Zstar =  Zstar + alpha(v) * Z{v};
                        end

                        final_Z = (abs(Zstar)+abs(Zstar'))/2;
                        grps = SpectralClustering(final_Z, clusters);
                    catch ME
                        fprintf(['ERROR: ' ME.message '\n'])
                        break
                    end
                    
                    time(m) = toc(tic1);

                   %% evaluation
                    P_label = bestMap(truth, grps);
                    ACCi(m) = length(find(truth == P_label))/length(truth);
                    [~, NMIi(m), ~] = compute_nmi(truth, grps);
                    [F1i(m), Pi(m), Ri(m)] = compute_f(truth,grps); 
                    [ARIi(m),~,~,~] = RandIndex(truth,grps);
                end
                ACC = mean(ACCi); std_ACC = std(ACCi);
                NMI = mean(NMIi); std_NMI = std(NMIi);
                P = mean(Pi); std_P = std(Pi);
                R = mean(Ri); std_R = std(Ri);
                F1 = mean(F1i); std_F1 = std(F1i);
                ARI = mean(ARIi); std_ARI = std(ARIi);
                
                output = [ACC NMI P R F1 ARI...
                        std_ACC std_NMI std_P std_R std_F1 std_ARI...
                        opts.lambda(1) opts.lambda(2) opts.lambda(3) ...
                        multiViewLLE_settings.K main_opts.normalize_method mean(time)];
                
                fprintf('\nACC: %0.4f(%0.4f)\n', ACC, std_ACC);
                fprintf('NMI: %0.4f(%0.4f)\n', NMI, std_NMI);
                fprintf('Precision: %0.4f(%0.4f)\n', P, std_P);
                fprintf('Recall: %0.4f(%0.4f)\n', R, std_R);
                fprintf('F1: %0.4f(%0.4f)\n', F1, std_F1);
                fprintf('ARI: %0.4f(%0.4f)\n', ARI, std_ARI);
                fprintf('time: %0.4f\n\n', mean(time));
                if main_opts.save_result == 1
                    dlmwrite(filename, output, '-append', 'precision',...
                        '%.4f', 'newline', 'pc' );
                end 
            end
        end
    end
end
end