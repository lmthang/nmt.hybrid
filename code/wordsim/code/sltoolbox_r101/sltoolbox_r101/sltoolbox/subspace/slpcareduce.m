function S = slpcareduce(S, cri, varargin)
%SLPCAREDUCE Reduces a PCA model to lower dimension
%
% $ Syntax $
%   - S = slpcareduce(S, cri, ...)
%
% $ Arguments $
%   - S:        the target PCA model
%   - cri:      the criterion for PCA model reduction
%
% $ Description $
%   - S = slpcareduce(S, cri, ...) reduces a PCA model by taking a subset
%     of principal components. cri can be a vector of selected indices of 
%     principal components, or a string representing the preset, 
%     reduction scheme, or a function handle of for selection. The 
%     user-supplied selection function takes a PCA model as input arg,
%     and return the indices of retained components. Additional arguments
%     are the feed to the criterion or user-selection function following
%     the PCA model.
%     \*
%     \t    Table 1.  The preset selection schemes   \\
%     \h     name     &      description             \\
%           'num'     & Select the first n components: use n as the 
%                       first argument.
%           'energy'  & Select the smallest number of components, so
%                       that the ratio of the preserved energy to the
%                       total energy is not less than the specified 
%                       ratio r, which is supplied as the first argument.
%                       Note that if the energy preservation of original
%                       PCA model is lower than r, an error will be raised.
%           'eigval'  & Select all components of which the ratio to the
%                       maximum eigenvalue is above a specified ratio r,
%                       which is supplied as the first argument.                        
%
% $ Remarks $
%   1. The energy of the discarded components in the reduction will be
%      added to the residue.
%
% $ History $
%   - Created by Dahua Lin on Apr 24, 2006
%   - Modified by Dahua Lin on Aug 17, 2006
%       - Add a field energyratio
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slpcareduce', 2);
end

inds_selected = [];  % the selected index
if isnumeric(cri)
    inds_selected = cri;
    
elseif ischar(cri)
    switch cri
        case 'num'
            fh_select = @pcselect_num;
        case 'energy'
            fh_select = @pcselect_energy;
        case 'eigval'
            fh_select = @pcselect_eigval;
        otherwise
            error('sltoolbox:invalidarg', ...
                'Invalid selection scheme %s for PCA reduction', cri);
    end
    
elseif isa(cri, 'function_handle')
    fh_select = cri;
    
else
    error('sltoolbox:invalidarg', ...
        'cri should be selected indices, a scheme name or the handle to the selection function');
end

%% select the components

if isempty(inds_selected)  % not selected yet    
    inds_selected = fh_select(S, varargin{:});        
end
RP = S.P(:, inds_selected);


%% reduce the PCA model according to the selection

S.feadim = length(inds_selected);
S.P = RP;

discarded_eigvals = S.eigvals;
discarded_eigvals(inds_selected) = [];
S.eigvals = S.eigvals(inds_selected);

S.residue = S.residue + sum(discarded_eigvals);

prinenergy = sum(S.eigvals);
S.energyratio = prinenergy / (prinenergy + S.residue);

%% Preset user selection criteria (scheme)

function inds = pcselect_num(S, n)

if n > S.feadim
    error('sltoolbox:valueexceed', ...
        'n is larger than the number of components preserved in the input model');
end

inds = (1:n)';

function inds = pcselect_energy(S, r)

pe = sum(S.eigvals);          % preserved energy in input model
te = pe + S.residue;            % total energy
ub_r = pe / te;               % upper bound on r

if r > ub_r
    error('sltoolbox:valueexceed', ...
        'r is larger than the ratio of energy preserved in the input model');
end

eb = te * r;
ce = cumsum(S.eigvals);
n = find(ce >= eb, 1);
if isempty(n)
    n = S.feadim;
end

inds = (1:n)';

function inds = pcselect_eigval(S, r)

if r < 0 || r >= 1
    error('sltoolbox:invalidarg', ...
        'It should be met that 0 < r <= 1 for eigenvalue ratio based selection');
end

n = sum(S.eigvals > r * S.eigvals(1));

inds = (1:n)';

        