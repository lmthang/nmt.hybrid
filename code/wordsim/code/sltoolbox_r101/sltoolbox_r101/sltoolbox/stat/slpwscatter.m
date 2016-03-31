function S = slpwscatter(tar, W)
%SLPWSCATTER Compute the pairwise scatter matrix
%
% $ Syntax $
%   - S = slpwscatter(X)
%   - S = slpwscatter({X, Y})
%   - S = slpwscatter(X, W)
%   - S = slpwscatter({X, Y}, W)
%
% $ Description $
%   - S = slpwscatter(X) computes the pairwise scatter matrix on the
%     samples X using the following formula. Suppose X has m samples
%     stored as columns, then 
%       S = sum_{i=1:m} sum_{j=1:m} X(:,i) * X(:,j)'
%
%   - S = slpwscatter({X, Y}) computes the pairwise scatter matrix on
%     samples X and Y using the following formula. Suppose X has m samples
%     and Y has n samples, then
%       S = sum_{i=1:m} sum_{j=1:n} X(:,i) * X(:,j)'
%
%   - S = slpwscatter(X, W) computes the weighted pairwise scatter matrix
%     on the samples X using the following formula. Suppose X has m
%     samples, then W should be an m x m matrix,
%       S = sum_{i=1:m} sum_{j=1:m} W(i,j) X(:,i) * X(:,j)'
%
%   - S = slpwscatter({X, Y}, W) computes the weighted pairwise scatter
%     matrix on the samples X using the following formula. Suppose X has
%     m samples and Y has n samples, then W should be an m x n matrix,
%       S = sum_{i=1:m} sum_{j=1:n} W(i,j) X(:,i) * Y(:,j)'
%
% $ Remarks $
%   - Instead of using the aforementioned formulas directly in computation,
%     it converts the problem into matrix multiplication, thus much more
%     efficient implementation can be applied.
%
% $ History $
%   - Created by Dahua Lin on Apr 27, 2006
%

%% parse and verify input arguments

% check target

if isnumeric(tar)   % X
    
    X = tar;
    if ndims(X) ~= 2
        error('sltoolbox:invaliddims', ...
            'X should be a 2D matrix');
    end
    m = size(X, 2);    
    
    targettype = 1;
    
elseif iscell(tar)  % {X, Y}
    
    if length(tar) ~= 2
        error('sltoolbox:invalidarg', ...
            'For cell target, it should be in the form {X, Y}');
    end
    
    X = tar{1};
    Y = tar{2};
    
    if ndims(X) ~= 2 || ndims(Y) ~= 2
        error('sltoolbox:invaliddims', ...
            'X and Y should be a 2D matrices');
    end
    
    if size(X, 1) ~= size(Y, 1)
        error('sltoolbox:sizmismatch', ...
            'The sample dimensions in X and Y are inconsistent');
    end
    
    m = size(X, 2);
    n = size(Y, 2);
    
    targettype = 2;
    
else
    
    error('sltoolbox:invalidarg', ...
        'The tar should be either X or {X, Y}');
    
end

% check weight

if nargin < 2
    W = [];
end

if ~isempty(W)    
    switch targettype        
        case 1
            if ~isequal(size(W), [m, m])
                error('sltoolbox:sizmismatch', ...
                    'The weight matrix W should be an m x m matrix');
            end            
        case 2
            if ~isequal(size(W), [m, n])
                error('sltoolbox:sizmismatch', ...
                    'The weight matrix W should be an m x n matrix');
            end            
    end    
end
        
        
%% compute

switch targettype
    
    case 1          % X
   
        if isempty(W)   % not weighted            
            D = diag(m(ones(m, 1)));
            W = ones(m, m);            
            M = 2 * (D - W);
            
            clear D W;
        else            % weighted
            D = diag(sum(W,1)' + sum(W,2));
            M = D - (W + W');
            
            clear D;
        end
        
        S = X * M * X';
                                                                                   
    case 2          % X, Y
        
        if isempty(W)   % not weighted            
            S1 = X * diag(m(ones(m, 1))) * X';
            S2 = Y * diag(n(ones(n, 1))) * Y';
            S12 = S1 + S2;
            clear S1 S2;
            
            S3 = X * ones(m, n) * Y';
            S34 = S3 + S3';
            clear S3;
            
            S = S12 - S34;            
        else            % weighted
            S1 = X * diag(sum(W, 2)) * X';
            S2 = Y * diag(sum(W, 1)) * Y';
            S12 = S1 + S2;
            clear S1 S2;
            
            S3 = X * W * Y';
            S34 = S3 + S3';
            clear S3;
            
            S = S12 - S34;            
        end
            
                            
end



    
    
    
    
    
    