function [Z, alpha] = CMSMSC(X, W, settings)

rho = 1.9;
mu = 0.1;
mu_max = 1e10;

if isfield(settings,'epsilon')
    epsilon = settings.epsilon;
else
    epsilon = 1e-6;
end

if isfield(settings,'max_iter')
    max_iter = settings.max_iter;
else
    max_iter = 50;
end

if isfield(settings,'display')
    display = settings.display;
    if display
        cost_trace = [];
        stop_trace = [];
        a_trace = [];
        b_trace = [];
        c_trace = [];
    end
else
    display = false;
end

Lstar = constructManifold(W);
lambda = settings.lambda;

n_v = numel(X);
alpha = ones(n_v, 1) * (1/n_v);
N = size(X{1}, 2);
I = eye(N);
I2 = eye(n_v);
B = zeros(n_v);
XTX = cell(n_v, 1);
Z = cell(n_v, 1);
Z_tmp = cell(n_v, 1);
XZ = cell(n_v, 1);
C = cell(n_v, 1);
E = cell(n_v, 1);
G1 = cell(n_v, 1);
G2 = cell(n_v, 1);
for v=1:n_v
    XTX{v} =  X{v}' * X{v};
    Z{v} = zeros(N);
    Z_tmp{v} = zeros(N);
    XZ{v} = zeros(size(X{v}));
    C{v} = zeros(N);
    E{v} = zeros(size(X{v}));
    G1{v} = zeros(size(X{v}));
    G2{v} = zeros(N);
end
iter = 0;
while iter < max_iter
    iter = iter + 1;
    
    % update C
    for v=1:n_v
       Q = Z{v} - G2{v}./mu;
       C{v} = sign(Q).*max(abs(Q)-1/mu, 0);
    end
    
    % update Z
    for v=1:n_v
        tmp = zeros(N);
        for i=1:n_v
            if i~=v
                tmp = tmp + alpha(v)*Z{v};
            end
        end
        Z{v} = (2*lambda(2)*alpha(v)*alpha(v).*Lstar + mu*XTX{v} + mu*I)\...
               (mu*XTX{v} - mu*X{v}'*E{v} + X{v}'*G1{v} ...
               - 2*lambda(2)*alpha(v)*Lstar*tmp + mu*C{v} + G2{v});
        Z{v} = Z{v} - diag(diag(Z{v}));
    end
    
    %update E
    for v=1:n_v
        XZ{v} = X{v} * Z{v};
        E{v} = mu/(2*lambda(1)-mu).*(XZ{v}-X{v}-G1{v}./mu);
    end
    
    %update alpha
    B = zeros(n_v);
    for i=1:n_v
        for j=i:n_v
            B(i,j) = reshape(Lstar'*Z{i}, [], 1)'*Z{j}(:);
%             B(i,j) = trace(Z{i}' * Lstar * Z{j});
        end
    end
    B = B + tril(B',-1);
    
    A = 2 * lambda(2) .* B + 2 * lambda(3) .* I2;
    gradient_function = @(x) A * x;
    opts.lipschitz_constant = max(abs(sum(gradient_function(alpha))));
    opts.length = n_v;
    opts.display = false;
    if opts.display
        fun = @(x) x' * A * x;
        alpha = exponentiated_gradient(gradient_function, opts, fun);
    else
        alpha = exponentiated_gradient(gradient_function, opts);
    end
    
    stop = 0;
    for i=1:n_v
        stop = max(max(...
            max(max(abs(X{v}-XZ{v}-E{v}))), max(max(abs(C{v} - Z{v})))),...
            stop);
    end
    
    if display
        [a, b, c] = objective(Z, E, alpha, Lstar, lambda);
        cost_trace = [cost_trace a+b+c]; %#ok<AGROW>
        a_trace = [a_trace a];
        b_trace = [b_trace a];
        c_trace = [c_trace a];
        stop_trace = [stop_trace stop];
    end
    
    if display && (iter==1 || mod(iter,50)==0 || stop < epsilon)
        fprintf(['\t+++ iter = ' num2str(iter) ...
            ', stop=' num2str(stop,'%2.6e') '\n']);
    end
    
    if stop < epsilon
        if display
            fprintf('=== core algorithm stop ===\n')
        end
        break;
    else
        for v=1:n_v
            G1{v} = G1{v} + mu * (X{v} - XZ{v} - E{v}); 
            G2{v} = G2{v} + mu * (C{v} - Z{v});
        end
        mu = min(rho*mu, mu_max);
    end
end
if display
    figure(1)
    plot(cost_trace, 'r.-', 'LineWidth', 2, 'MarkerSize', 20)
    a = get(gca,'XTickLabel');
    set(gca, 'XTickLabel',a ,'fontsize',14)
    grid on
    xlabel('# of Iterations', 'FontSize', 15)
    ylabel('Objective Value', 'FontSize', 15)
end
end

function Lstar = constructManifold(W)
    Astar = 0.5*(abs(W)+abs(W'));
    Dstar = diag(sum(Astar, 2));
    Lstar = Dstar - Astar;
end

function [a, manifold, c]=objective(Z, E, alpha, Lstar, lambda)
    a = 0;
    n_v = numel(Z);
    Zstar = zeros(size(Z{1}));
    for v=1:n_v
        a1 = sum(sum(abs(Z{v})));
        a2 = norm(E{v}, 'fro')^2;
        a = a + a1 + lambda(1) * a2;
        Zstar =  Zstar + alpha(v) * Z{v};
    end
%     manifold = lambda(2) * trace(Zstar'*Lstar*Zstar);
    manifold = lambda(2) * reshape(Lstar'*Zstar, [], 1)'*Zstar(:);
    c = lambda(3) * (alpha'*alpha);
end