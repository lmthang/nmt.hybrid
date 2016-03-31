function d = sldim_by_eigval(eigvals, sch, varargin)
%SLDIM_BY_EIGVAL Determines the dimension of principal subspace by eigenvalues
%
% $ Syntax $
%   - d = sldim_by_eigval(eigvals)
%   - d = sldim_by_eigval(eigvals, sch, ...)
%
% $ Arguments $
%   - eigvals:      the eigenvalues (energies) of each dimension given in
%                   descending order
%   - sch:          the name of scheme of dimension evaluation, or a
%                   function handle to some user-supplied functions, which
%                   takes the list of eigenvalues as the first argument.
%
% $ Description $
%   - d = sldim_by_eigval(eigvals) determines the dimension of the
%     principal subspace according to eigenvalues in default way:
%     'rank', i.e. set the dimension to the rank.
%
%   - d = sldim_by_eigval(eigvals, sch, ...) determines the dimension of
%     the principal subspace according to eigenvalues using a specified
%     scheme. The additional arguments for the parameters for the scheme.  
%     \*
%     \t    Table 1. The available schemes of dimension evaluation \\
%     \h      name    &          description                  \\
%            'rank'   &  The dimension is determined up to the rank
%                        implied by the eigenvalues. In detail, d is
%                        the number of eigenvalues larger than the
%                        eps(max(eigvals))  \\
%            'ratio'  &  The dimension is determined by the number of
%                        eigenvalues that is larger than r * max(eigvals).
%                        Here r is given as the first scheme argument. \\
%            'energy' &  The dimension is determined by the smallest number
%                        of leading eigenvalues, so that the ratio of their 
%                        sum (i.e. the energy preserved in the principal 
%                        subspace) to the total sum is not less than r,
%                        which is given as the first scheme argument. \\
%     \*
%
% $ Remarks $
%   - It is a required condition that the eigenvalues are not non-negative
%     and sorted in descending order.
%                        
% $ History $
%   - Created by Dahua Lin on Apr 25, 2006
%


%% parse and verify input arguments

if nargin < 2
    sch = 'rank';
end

% determine the function

if ischar(sch)
    
    switch sch
        case 'rank'
            fh = @dim_by_rank;
        case 'ratio'
            fh = @dim_by_ratio;
        case 'energy'
            fh = @dim_by_energy;
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid scheme %s for dimension determination', sch);
    end
    
elseif isa(sch, 'function_handle')
    fh = sch;
    
else
    error('sltoolbox:invalidarg', ...
        'Invalid sch: it should be either the name of the scheme or the user-supplied function handle');
end

%% select

d = fh(eigvals, varargin{:});

%% The predefined selection function


function k = dim_by_rank(evals)

k = sum(evals > eps(evals(1)));

function k = dim_by_ratio(evals, r)

if (r <= 0 || r >= 1)
    error('r should satisfy 0 < r < 1 for preservation by eigval');
end

k = sum(evals > r * evals(1));

function k = dim_by_energy(evals, r)

if (r <= 0 || r >= 1)
    error('r should satisfy 0 < r < 1 for preservation by energy');
end

energies = cumsum(evals);
k = find(energies >= r * energies(end), 1);


