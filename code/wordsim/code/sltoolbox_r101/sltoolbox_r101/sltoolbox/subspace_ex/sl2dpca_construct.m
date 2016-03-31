function X = sl2dpca_construct(Mm, PL, PR, Y)
%SL2DPCA_CONSTRUCT Constructs the matrix from 2D feature
%
% $ Syntax $
%   - X = sl2dpca_construct(Mm, PL, PR, Y)
%
% $ Arguments $
%   - Mm:       the mean matrix
%   - PL:       the left projection matrix
%   - PR:       the right projection matrix
%   - Y:        the extracted 2D features
%   - X:        the constructed matrices
%
% $ Description $
%   - X = sl2dpca_construct(Mm, PL, PR, Y) constructs the matrices in
%     original size using a 2D PCA model characterized by mean matrix and
%     the left and right projection matrices. 
%
% $ History $
%   - Created by Dahua Lin, on Jul 31st, 2006
%

%% Parse and verify input arguments

if ndims(Mm) ~= 2
    error('sltoolbox:invalidarg', ...
        'Mm should be a 2D matrix');
end
[d1, d2] = size(Mm);
if size(PL, 1) ~= d1 || size(PR, 1) ~= d2
    error('sltoolbox:sizmismatch', ...
        'Inconsistent size for 2D PCA model');
end
k1 = size(PL, 2);
k2 = size(PR, 2);
if size(Y, 1) ~= k1 || size(Y, 2) ~= k2
    error('sltoolbox:sizmismatch', ...
        'The feature size is inconsistent with the 2D PCA model');
end

%% Construct

n = size(Y, 3);
X = zeros(d1, d2, n);
PRT = PR';

for i = 1 : n
    X(:,:,i) = PL * Y(:,:,i) * PRT + Mm;
end


