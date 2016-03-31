function Y = sl2dpca_apply(Mm, PL, PR, data, matsiz, n)
%SL2DPCA_APPLY Applies 2D PCA onto a set of matrices to extract features
%
% $ Syntax $
%   - Y = sl2dpca_apply(Mm, PL, PR, data, matsiz, n)
%
% $ Description $
%   - Mm:           the mean matrix
%   - PL:           the left projection matrix
%   - PR:           the right projection matrix
%   - data:         the matrix samples or the cell array of filenames
%   - matsiz:       the original matrix size
%   - n:            the number of samples
%   - Y:            the extracted 2D features
%
% $ Description $
%   - Y = sl2dpca_apply(data, Mm, PL, PR) extracts 2D features for 
%     the matrices given in data, in either a 3D array or a set of
%     array filenames. Suppose the original matrix size is d1 x d2,
%     PL be d1 x k1, PR be d2 x k2, then the feature matrix would be
%     of size k1 x k2. Y is a k1 x k2 x n array.
%
% $ History $
%   - Created by Dahua Lin, on Jul 31st, 2006
%

%% Parse and verify input arguments

if nargin < 6
    raise_lackinput('sl2dpca_apply', 6);
end

matsiz = matsiz(:)';
if length(matsiz) ~= 2
    error('sltoolbox:invalidarg', ...
        'matsiz should be a 2-elem vector');
end

if ~isequal(size(Mm), matsiz)
    error('sltoolbox:sizmismatch', ...
        'the sample size does not match the model');
end
d1 = matsiz(1);
d2 = matsiz(2);

if size(PL, 1) ~= d1 || size(PR, 1) ~= d2
    error('sltoolbox:sizmismatch', ...
        'the size of projection matrices are illegal');
end

%% Compute

if isnumeric(data)
    
    if size(data, 3) ~= n
        error('sltoolbox:sizmismatch', ...
            'The number of samples is not as specified');
    end
    
    Y = computeY(data, Mm, PL, PR);
    
elseif iscell(data)
    
    Y = zeros(size(PL, 2), size(PR, 2), n);
    
    nfiles = length(data);
    cf = 0;
    for i = 1 : nfiles
        curdata = slreadarray(data{i});
        curn = size(curdata, 3);
        Y(:,:,cf+1:cf+curn) = computeY(curdata, Mm, PL, PR);
        cf = cf + curn;
    end
    
else
    error('sltoolbox:invalidarg', ...
        'data should be a numeric array or a cell array of filenames');    
 
end


%% Core compute function

function Y = computeY(X, Mm, PL, PR)

n = size(X, 3);
Y = zeros(size(PL, 2), size(PR, 2), n);
PLT = PL';

for i = 1 : n
    Y(:,:,i) = PLT * (X(:,:,i) - Mm) * PR;
end





    




