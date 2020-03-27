function X_transform = normalize(X, method)

% 每列2范数为1
if strcmp(method, 'L2')
    % X_transform  =  X ./ repmat(sqrt(sum(X.*X)), [size(X, 1),1]);
    X_transform = zeros(size(X));
    for i = 1:size(X, 2)
        X_transform(:,i) = X(:,i) ./ max(1e-12,norm(X(:,i)));
    end
end

% 每类归一化到[0,1]
if strcmp(method, 'MinMax')
    X_transform = rescale(X, 'InputMin', min(X), 'InputMax', max(X));
end

% 每列无穷范数为1
if strcmp(method, 'Inf')
    X_transform = X ./ max(abs(X));
end

end

