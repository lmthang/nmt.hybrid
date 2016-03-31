function S = sllabeledsum(X, labels, labelset, w)
%SLLABELEDSUM Sums the numbers according to labels
%
% $ Syntax $
%   - S = sllabledsum(X, labels, labelset)
%   - S = sllabledsum(X, labels, labelset, w)
%
% $ Arguments $
%   - X:            The matrix of numbers (m x n)
%   - labels:       The labels of columns in X (1 x n)
%   - labelset:     The set of labels used (length-c vector)
%   - w:            The column weights (1 x n) (default = [])
%   - S:            The sum (m x c)
%
% $ Description $
%   - S = sllabledsum(X, labels, labelset) sums the columns in X that 
%     corresponding to the same label. If labelset has c labels, then 
%     S would have c columns, the i-th column sums the columns in X 
%     associative the labelset(i).
%
%   - S = sllabledsum(X, labels, labelset, w) performs a weighted sum.
%
% $ History $
%   - Created by Dahua Lin, on Aug 31, 2006
%   - Modified by Dahua Lin, on Sep 10, 2006
%       - replace slmul by slmulvec to increase efficiency
%

%% parse and verify input 

if nargin < 3
    raise_lackinput('sllabeledsum', 3);
end

if ndims(X) ~= 2 || ~isnumeric(X)
    error('sltoolbox:invalidarg', 'X should be a numeric 2D matrix');
end
[m, n] = size(X);

if ~isequal(size(labels), [1 n])
    error('sltoolbox:sizmismatch', ...
        'The size of labels does not match the number of X-columns');
end

if ~isvector(labelset)
    error('sltoolbox:invalidarg', 'labelset should be a vector');
end
c = length(labelset);

if nargin < 4
    w = [];
else
    if ~isempty(w)
        if ~isequal(size(w), [1 n])
            error('sltoolbox:sizmismatch', ...
                'The size of w does not match the number of X-columns');
        end
    end
end

%% compute

Inds = sllabelinds(labels, labelset);
S = zeros(m, c);


for i = 1 : c
    curinds = Inds{i};
    if ~isempty(curinds)
        curX = X(:, curinds);        
        if ~isempty(w)
            curw = w(curinds);
            curX = slmulvec(curX, curw, 2);
        end        
        S(:,i) = sum(curX, 2);
    end
end



    



