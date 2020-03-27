function [weight, neighbours] = multiViewLLE(X, settings)
K = settings.K;

if isfield(settings,'epsilon')
    epsilon = settings.epsilon;
else
    epsilon = 1e-6;
end

if isfield(settings,'lle_max_iter')
    max_iter = settings.lle_max_iter;
else
    max_iter = 200;
end

if isfield(settings,'display')
    display = settings.display;
else
    display = false;
end

if isfield(settings,'gamma')
    gamma = settings.gamma;
else
    gamma = 1;
end

if isfield(settings,'eta')
    opts.eta = settings.eta;
else
    opts.eta = 0.1;
end

n_v = numel(X);
l = n_v * K;
N = size(X{1}, 2);
W = ones(l, N) ./ l;
g_tild = zeros(n_v, 1);

%% 多视图邻近点
% 按列排，一个点对应 l 个邻近点
distance = zeros(N, N);
for v = 1:n_v
    X2 = sum(X{v}.^2,1);
    tmp = repmat(X2, N, 1) + repmat(X2', 1, N) - 2*X{v}'*X{v};
    distance = distance + tmp./max(tmp);
end
[~, index] = sort(distance);
neighbours = index(2:(1+l), :); % 排除自己

%% 计算G
G = cell(N, n_v);
for i=1:N
    for v=1:n_v
        tmp = (X{v}(:,i) - X{v}(:, neighbours(:, i)));
        G{i,v} = tmp' * tmp;
    end
end

%% main loop
for i=1:N
    rho = ones(n_v, 1) * (1/n_v);
    iter = 0;
    if display
        loss = [];
        loss1 = [];
        loss2 = [];
    end
    while iter < max_iter
        iter = iter + 1;
        
        rho_prev = rho;
        w_prev = W(:, i);
        
        % update w_i
        G_hat = zeros(l);
        for v=1:n_v
            G_hat = G_hat + rho(v).*G{i,v};
        end
%         fun = @(x) x' * G_hat * x;
        gradient_function = @(x) 2 * G_hat * x;
        opts.lipschitz_constant = 2 * max(sum(G_hat));
        opts.length = l;
        opts.maxIter = 1000;
        opts.display = false;
        W(:,i) = exponentiated_gradient(gradient_function, opts);

        % update rho
        for v=1:n_v
            g_tild(v) = W(:,i)'*G{i,v}*W(:,i);
        end
        
        % by projection over simplex
        rho = proj_simplex(-g_tild/gamma, 1);

        if display
            [a, b] = objective(W, rho_prev, gamma, i, G);
            loss = [loss a+b]; %#ok<*AGROW>
            loss1 = [loss1 a];
            loss2 = [loss2 b];
        end
        
        % check stop condition
        stop = max(max(abs(W(:,i)-w_prev)), max(abs(rho-rho_prev)));
        
        if display
            fprintf(['\t+++ iter = ' num2str(iter) ...
                ', stopLoop=' num2str(stop,'%2.8e') '\n']);
        end
        if stop < epsilon
            break;
        end
    end
    if display
        plot(loss, 'r')
%         hold on
%         plot(loss1, 'g')
%         plot(loss2, 'b')
%         hold off
        title(['i=' num2str(i) ' ,loss=' num2str(loss(end))])
        pause(1)
    end
end

weight = zeros(N, N);
for i=1:N
    weight(neighbours(:,i), i) = W(:, i);
end

end

function [tmp, b] = objective(W, rho, gamma, i, G)
    n_v = length(rho);
    tmp = 0;
    for v=1:n_v
        tmp = tmp + rho(v) * (W(:,i)' * G{i,v} * W(:,i));
    end
    b = gamma * (rho' * rho);
end

