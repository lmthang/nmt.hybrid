function [X, spectrum] = slcmds(D, d, w, ty)
%SLMDS Performs Classical Multidimensional scaling
%
% $ Syntax $
%   - X = slcmds(D, d)
%   - X = slcmds(D, d, w)
%   - X = slcmds(D, d, w, 'sqr')
%   - [X, spectrum] = slcmds(...)
%
% $ Arguments $
%   - D:        The pairwise distance matrix (n x n)
%   - d:        The dimension of the embedding space
%   - w:        The weights of samples (1 x n or [])
%   - X:        The embedded samples (d x n)
%
% $ Description $
%   - X = slcmds(D, d) performs classic multidimensional scaling to
%     pursue an embedding space of d-dimension and the vector 
%     representation in that space of the objects, such that the 
%     distances are optimally preserved.
%
%   - X = slcmds(D, d, w) If w is not empty, it performs classic 
%     multidimensional scaling on weighted samples. 
%
%   - X = slcmds(D, d, w, 'sqr') indicates that D contains the square
%     of distances.
%
%   - [X, spectrum] = slcmds(...) additionally outputs the spectrum of
%     the embedded space
%     
% $ History $
%   - Created by Dahua Lin, on Sep 8th, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slcmds', 2);
end

if ndims(D) ~= 2 || size(D, 1) ~= size(D, 2)
    error('sltoolbox:invalidarg', ...
        'The D should be a square matrix');
end
n = size(D, 1);

if d >= n
    error('sltoolbox:exceedbound', ...
        'The dimension d should be less than the number of samples n');
end

if nargin < 3
    w = [];
else
    if ~isempty(w)
        if ~isequal(size(w), [1, n])
            error('sltoolbox:sizmismatch', ...
                'If w is specified, it should be an 1 x n row vector');
        end
    end
end

if nargin >= 4 && strcmpi(ty, 'sqr')
    is_sqr = true;
else
    is_sqr = false;
end


%% compute

if ~is_sqr
    K = sldists2kernels(D);
else
    K = sldists2kernels(D, 'sqr');
end

[X, spectrum] = slkernelembed(K, d, w);





    
    


