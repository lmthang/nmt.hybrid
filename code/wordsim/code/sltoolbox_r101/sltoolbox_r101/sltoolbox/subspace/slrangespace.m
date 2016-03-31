function [U, Uc] = slrangespace(tar, varargin)
%SLRANGESPACE Determines the subspace of the range of X
%
% $ Syntax $
%   - U = slrangespace(tar)
%   - U = slrangespace(tar, ...)
%   - [U, Uc] = slrangespace(...)
%
% $ Arguments $
%   - tar:      the target, can be a sample matrix or covariance
%   - U:        the basis for the range space
%   - Uc:       the basis for the orthogonal complement of the range
%
% $ Description $
%   - U = slrangespace(tar) determines the range space of tar in default 
%     settings. That is to set the dimension to be the rank of tar. Note
%     that tar can have two forms: a sample matrix or a covariance
%     given by the syntax {'cov', C}.
%
%   - U = slrangespace(tar, ...) determines the range space of X using
%     the specified dimension determination schemes. The arguments 
%     input following X will be delivered to sldim_by_eigval for dimension
%     determination.
%
%   - [U, Uc] = slrangespace(...) also returns orthogonal complement of U.
%
% $ History $
%   - Created by Dahua Lin on Apr 25, 2005
%

%% parse and verify input arguments

if isnumeric(tar)   
    target_type = 1;        % target is sample
    
    if ndims(tar) ~= 2
        error('sltoolbox:invalidarg', ...
            'When tar is sample matrix, it should be a 2D matrix');
    end
    
    X = tar;
    d = size(X, 1);
    
elseif iscell(tar)
    
    if length(tar) == 2 && ischar(tar{1}) ...
            && strcmpi(tar{1}, 'cov')
        
        target_type = 2;        % target is covariance
        
        C = tar{2};
        d = size(C, 1);
        
        if ~isequal(size(C), [d d])
            error('sltoolbox:invalidarg', ...
                'A covariance matrix should be square');
        end
        
    else        
        error('sltoolbox:invalidarg', ...
            'The target should be a sample matrix or a covariance given by cell');        
    end
    
else
    
    error('sltoolbox:invalidarg', ...
        'The target should be a sample matrix or a covariance given by cell');        
end

if nargout == 0
    return;
elseif nargout == 1
    need_complement = false;
else
    need_complement = true;
end
    

%% compute

switch target_type
    
    case 1      % sample matrix     
        
        if ~need_complement
            [U, D] = svd(X, 0);
        else
            [U, D] = svd(X);
        end
        
        evals = diag(D) .^ 2;
        clear D;
                                        
    case 2      % covariance
        
        [evals, U] = slsymeig(C);
        
end
    
evals = max(evals, 0);
k = sldim_by_eigval(evals, varargin{:});

%% output

if ~need_complement
    U = U(:, 1:k);
else
    if k == size(U, 2)
        Uc = zeros(d, 0);
    else
        Uc = U(:, k+1:end);
        U  = U(:, 1:k);
    end
end

