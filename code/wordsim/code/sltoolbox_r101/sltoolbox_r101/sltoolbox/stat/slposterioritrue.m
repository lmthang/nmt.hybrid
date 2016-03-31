function posteriori = slposterioritrue(condprops, nums, priori, op)
%SLPOSTERIORITRUE Computes the posteriori that samples belong to true class
%
% $ Syntax $
%   - posteriori = slposterioritrue(condprops, nums, priori)
%   - posteriori = slposterioritrue(condprops, nums, priori, 'log')
%
% $ Arguments $
%   - condprods:      the conditional probabilities of classes: C x n
%   - nums:           the number of samples belong to the classes: 1 x C
%   - priori:         the prior probabilities of classes: 1 x C
%   - posteriori:     the resulting posterior probabilities: 1 x n
%
% $ Description $
%   - posteriori = slposteriori(condprops, nums, priori) Computes the 
%     posterior probability that the samples belong to the true class 
%     according to the given conditional probabilities of all samples 
%     to all classes and the priori of the classes. 
%     In the input argument, each column in condprops represent the 
%     conditional probabilities of that sample belong to all the 
%     C classes. The samples from the same underlying classes should be
%     put together and the ownership is given by nums.
%     The priori can be given by an 1 x C row vector or [], which 
%     means that all classes have equal prior.
%
%   - posteriori = slposterioritrue(condprops, nums, priori, 'log') means
%     that the input condprops are given by its logarithm.
%
% $ History $
%   - Created by Dahua Lin, on Sep 16th, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slposterioritrue', 2);
end

if ~isnumeric(condprops) || ndims(condprops) ~= 2
    error('sltoolbox:invalidarg', ...
        'The condprops should be a 2D numeric matrix');
end
[C, n] = size(condprops);

if ~isequal(size(nums), [1, C])
    error('sltoolbox:sizmismatch', ...
        'The nums should be an 1 x C row vector');
end
if nargin < 3 || isempty(priori)
    priori = [];
else
    if ~isequal(size(priori), [1, C])
        error('sltoolbox:sizmismatch', ...
            'The priori should be a an 1 x C row vector');
    end
end

if nargin >= 4 && strcmpi(op, 'log')
    is_log = true;
else
    is_log = false;
end

%% compute

if is_log
    if isempty(priori)
        P = sladdvec(condprops, -max(condprops, [], 1), 2);
    else
        P = sladdvec(condprops, log(priori)', 1);
        P = sladdvec(P, -max(condprops, [], 1), 2);
    end
    P = exp(P);
else
    if isempty(priori)
        P = condprops;
    else
        P = slmulvec(condprops, priori', 1);
    end
end

tP = sum(P, 1);
[sp, ep] = slnums2bounds(nums);
posteriori = zeros(1, n);

for k = 1 : C
    sk = sp(k);
    ek = ep(k);
    posteriori(sk:ek) = P(k, sk:ek);
end
posteriori = posteriori ./ tP;

