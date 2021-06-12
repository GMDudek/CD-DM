function [Yp] = CDDMpredict(a, b, beta, X)
%CDDMpredict - prediction function for CCDM
%
% a - hidden node weights, n x m
% b - hidden node biases, 1 x m
% beta - output weights, m x 1
% X - inputs, n x N, n - #features, N - #samples

m = length(b);
[n,N] = size(X);
h = nan(N,m);

for i=1:m
    h(:,i) = 1 ./ (1 + exp(-(a(:,i)' * X + b(i)))); %hidden layer output
end

fr = h .* repmat(beta',N,1);
Yp = sum(fr,2); %predicted output

end