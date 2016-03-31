function posteriori = slposteriori(condprops, priori, op)
%SLPOSTERIORI Computes the posterioris 
%
% $ Syntax $
%   - posteriori = slposteriori(condprops, priori)
%   - posteriori = slposteriori(condprops, priori, 'log')
%
% $ Arguments $
%   - condprods:      the conditional probabilities of classes
%   - priori:         the prior probabilities of classes
%   - posteriori:     the resulting posterior probabilities
%
% $ Description $
%   - posteriori = slposteriori(condprops, priori) Computes the posterior
%     probability according to the given conditional probabilities of all
%     samples to all classes and the priori of the classes. If the number
%     of classes is C and the number of samples is n, then the size of 
%     condprops should be k * n, the size of priori should be k-dim vector. 
%     And the resultant posteriori matrix will be of size k * n.
%
%   - posteriori = slposteriori(condprops, priori, 'log') where condprops
%     are given by its logarithm. And the computation is based on logarithm
%     in a stable manner.
%
% $ Remarks $
%   - If priori is not specified, then they are assumed to be equal.
%
% $ History $
%   - Created by Dahua Lin on Dec 21st, 2005
%   - Modified by Dahua Lin on Apr 22, 2006
%       - fix some header comments
%       - fix some places to increase efficiency
%   - Modified by Dahua Lin on Sep 10, 2006
%       - replace sladd by sladdvec and slmul by slmulvec to increase 
%         efficiency.
%

%% parse and verify
k = size(condprops, 1);
if nargin < 2 || isempty(priori)
    priori = ones(k, 1) / k;
else
    if numel(priori) ~= k
        error('sltoolbox:sizmismatch', ...
            'The size of priori is not consisitent with that of condprops');
    end
    priori = priori(:);
end
if nargin < 3 || isempty(op)
    logstyle = false;
else 
    if strcmpi(op, 'log')
        logstyle = true;
    else
        error('sltoolbox:invalidoption', ...
            'Invalid option %s', op);
    end
end

%% compute
if logstyle
    prods = sladdvec(condprops, log(priori), 1);
    prods = sladdvec(prods, -max(prods, [], 1), 2);
    prods = exp(prods);
else
    prods = slmulvec(condprops, priori);
end

tprods = sum(prods, 1);
posteriori = slmulvec(prods, 1 ./ tprods);



    