function [a, b, beta] = CDDM(X, Y, m, k, theta, Q, q_max, RMSE_max)
%CDDM - constructive data-driven method for randomized learning of FNNs
%
% X - inputs, n x N, n - #features, N - #samples
% Y - target outputs, N x 1
% m - number of hidden nodes
% k - number of nearest neighbours, k>=n
% theta - initial value of the threshold for the error change, theta<=0
% Q - threshold for adaptation of theta
% q_max - maximum number of the consecutive unaccepted candidate nodes for stop condition
% RMSE_max - maximum allowable error for stop condition
% a - hidden node weights, n x m
% b - hidden node biases, 1 x m
% beta - output weights, m x 1

[n,N] = size(X);

a = nan(n,m);
b = nan(1,m);
h = nan(N,m);
rmse = nan(1,m);
drmse = nan(1,m);

d = dist(X); %distance between input points
[~,is] = sort(d); %indices of the nearest neighbours

i = 1; l = 1; q = 1; rmse1 = RMSE_max + 1;
while (i <= m) && (rmse1 > RMSE_max) && (q < q_max) %main loop
    
    if rem(l,N) == 1
        ig = randperm(N,m); %choose randomly x*-points
        xg = X(:,ig); %x*
        ik = is(1:k,ig); %indices of the k nearest neighbours of x*
    end
    
    xp = [ones(k,1) X(:,ik(:,i))'];
    yp = Y(ik(:,i));
    xp = xp + rand(size(xp)) * 1e-10; %to avoid numerical errors
    ap = xp \ yp; %hyperplane fitting to the neighborhood
    
    a(:,i) = 4 * ap(2:end); %hidden node weights
    b(i) = -a(:,i)' * xg(:,i); %hidden node bias
    h(:,i) = 1 ./ (1 + exp(-(a(:,i)' * X + b(i)))); %hidden layer output
    
    beta = pinv(h(:,1:i)) * Y; %output weights
    
    fr = h(:,1:i) .* repmat(beta',N,1);
    Y1 = sum(fr,2); %predicted output
    rmse(i) = (mean((Y1 - Y).^2))^0.5; %RMSE
    
    if i > 1
        drmse(i) = rmse(i) - rmse(i-1); %error change
    end
    q = q + 1;
    
    %add node if i == 1 or error reduction over theta
    if (i == 1) || (drmse(i) <= theta)
        rmse1 = rmse(i);
        i = i + 1;
        q=1;
    end
    
    if rem(q,Q)==0
        theta = theta / 2; %theta is halved
    end
    
    l = l + 1; %main loop counter
end
end