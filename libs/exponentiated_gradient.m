function x = exponentiated_gradient(gradient_function, settings, fun)

if exist('fun', 'var') 
    loss = [];
    calc_cost = true;
else
    calc_cost = false;
end

if isfield(settings,'epsilon')
    epsilon = settings.epsilon;
else
    epsilon = 1e-5;
end

if isfield(settings,'maxIter')
    max_iter = settings.maxIter;
else
    max_iter = 500;
end

if isfield(settings,'display')
    display = settings.display;
else
    display = false;
end

if isfield(settings,'lipschitz_constant')
    inv_lip_constant = 1/settings.lipschitz_constant;
else
    if isfield(settings,'eta')
        eta = settings.eta;
    else
        eta = 0.1;
    end
end

d = settings.length;

iter = 0;
x = ones(d, 1) * (1/d);

while iter < max_iter
    iter = iter + 1;
    gradient = gradient_function(x);
    
    if isfield(settings,'lipschitz_constant')
        eta = sqrt(2*log(d))*inv_lip_constant*(1/sqrt(iter));
    end
    
    Z = x.*exp(-eta.*gradient);
    
    x_prev = x;
    x = Z ./ sum(Z);
   
    if calc_cost
        loss = [loss fun(x)];
    end
    stop = max(abs(x-x_prev));% 或者用梯度值判断
    
    if display && (iter==1 || mod(iter,50)==0 || stop<epsilon)
        fprintf(['\t Exponentiated Gradient iter = ' num2str(iter) ...
                ', stop=' num2str(stop,'%2.8e') '\n']);
    end
    
    if stop < epsilon
        break
    end
end
if display && calc_cost
    plot(loss, '.-')
    1
end
end