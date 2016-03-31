function H = slcountvote(m, n, V, w, countrule)
%SLCOUNTRULE Counts the votings to make histogram
%
% $ Syntax $
%   - H = slcountvote(m, n, V, w, countrule)
%
% $ Arguments $
%   - m:            The number of models to be voted for
%   - n:            The number of samples
%   - V:            The voting results
%   - w:            The weights of the samples
%   - countrule:    The rule of counting (summarizing the votings)
%
% $ Description $
%   - H = slcountvote(m, n, V, w, countrule) counts/summarizes the 
%     voting results using specified rule to build a histogram.
%     There are m models, and the H would be an m x 1 column vector.
%     The following rules are implemented. Please note that using
%     different ways, the format of V will be different.
%     \*
%     \t    Table. The Couting rules in slvote
%     \h     name      &          description
%            'nm'      & (Nearest Model): the simplest way of voting. Each 
%                        sample contributes 1 to the nearest model.
%                        V is an 1 x n row vector indicating the indices
%                        of the matched models for all samples.
%            'nmx'     & (Extended Nearest Model): each sample contributes
%                        to the nearest model with a value indicating the
%                        confidence.
%                        V is an 2 x n matrix with the first row indicating
%                        the indices of the matched models, while the 2nd
%                        row indicates the confidence values.
%            'mmc'     & (Multi-Model Count): each sample contributes to 
%                        multiple models, the contribution to each model
%                        is 1.
%                        V is an m x n matrix or sparse matrix. The entry
%                        V(k,i) is non-zero when the i-th sample
%                        contributes to the k-th model.
%            'mms'     & (Multi-Model Sum): each sample contributes to 
%                        multiple models with real values. The voting on
%                        each model is summarized by summing the
%                        contribution values from all samples.
%                        V is an m x n matrix.
%            'mmns'    & (Multi-Model Normalized Sum): each sample 
%                        contributes to multiple models with real values.
%                        The voting on each model is summarized by summing
%                        the contribution values from all samples. 
%                        However, in contrary to mms, the total
%                        contributions from a sample would be normalized 
%                        to one. V is an m x n matrix.
%     \* 
% 
% $ History $
%   - Created by Dahua Lin, on Sep 18, 2006
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('slcountvote', 5);
end

if ~isempty(w)
    if ~isequal(size(w), [1 n])
        error('sltoolbox:sizmismatch', ...
            'The weights size is not consistent with that of sample number');
    end
end

% count voting and make histogram

switch countrule
    case 'nm'
        H = countvote_nm(V, w, m, n);
    case 'nmx'
        H = countvote_nmx(V, w, m, n);
    case 'mmc'
        H = countvote_mmc(V, w, m, n);
    case 'mms'
        H = countvote_mms(V, w, m, n);
    case 'mmns'
        H = countvote_mmns(V, w, m, n);
    otherwise
        error('sltoolbox:invalidarg', ...
            'Invalid counting rule: %s', countrule);
end


%% Vote-Count functions

function H = countvote_nm(V, w, m, n)

check_Vsize('Nearest Model', V, 1, n);

if isempty(w)
    H = zeros(m, 1);
    [nums, u] = slcount(sort(V));
    H(u) = nums;
else
    H = sllabeledsum(w, V, 1:m)';
end


function H = countvote_nmx(V, w, m, n)

check_Vsize('Extended Nearest Model', V, 2, n);

H = sllabeledsum(V(2,:), V(1,:), 1:m, w);


function H = countvote_mmc(V, w, m, n)

check_Vsize('Multi-Model Count', V, m, n);

if isempty(w)
    H = sum(V ~= 0, 2);
else
    H = (V ~= 0) * w';
end
if issparse(H)
    H = full(H);
end


function H = countvote_mms(V, w, m, n)

check_Vsize('Multi-Model Sum', V, m, n);

if isempty(w)
    H = sum(V, 2);
else
    H = V * w';
end
if issparse(H)
    H = full(H);
end


function H = countvote_mmns(V, w, m, n)

check_Vsize('Multi-Model Normalized Sum', V, m, n);

ss = sum(V, 1);
if issparse(ss)
    ss = full(ss);
end
ss(ss < eps) = eps;

if isempty(w)
    w = 1 ./ ss;
else
    w = w ./ ss;
end

H = V * w';
if issparse(H)
    H = full(H);
end


%% Auxiliary functions

function check_Vsize(rulename, V, d1, d2)

if ~isequal(size(V), [d1, d2])
    error('sltoolbox:invalidarg', ...
        'V should be an %d x %d matrix for the counting rule %s', ...
        d1, d2, rulename);
end

