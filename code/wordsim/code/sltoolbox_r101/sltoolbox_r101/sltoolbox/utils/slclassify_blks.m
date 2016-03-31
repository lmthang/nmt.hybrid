function [decisions, decscores] = slclassify_blks(scores, n, blocks, clabels, op, varargin)
%SLCLASSIFY_BLKS Classifies samples according to blockwise scores
%
% $ Syntax $
%   - [decisions, decscores] = slclassify_blks(scores, n, blocks, clabels, op, ...)
%
% $ Arguments $
%   - scores:       the score matrix
%   - n:            the number of query samples
%   - blocks:       the cell array of block limits
%   - clabels:      the class labels of reference samples
%   - op:           the score attribute  
%   - decisions:    the classification decisions
%   - decscores:    the scores of the classified targets
%
% $ Remarks $
%   - An extension of slclassify to support blockwise scores.
%
% $ History $
%   - Created by Dahua Lin on Aug 9th, 2006
%   - Modified by Dahua Lin, on Aug 16th, 2006
%       - eliminate the qlabel parameters, which are essentially not
%         needed.
%       - add functionality to support schemes. In current revision,
%         it supports nearest-neighbor ('nn') and leave-one-out ('loo').
%

%% parse and verify input arguments

if nargin < 5
    raise_lackinput('slclassify_blks', 5);
end
if ~iscell(scores)
    error('sltoolbox:invalidarg', ...
        'The scores should be a cell array of filenames');
end
if ~isequal(size(scores), size(blocks))
    error('sltoolbox:sizmismatch', ...
        'The sizes of scores and blocks are inconsistent');
end
[nrows, ncols] = size(blocks);
if ~ismember(op, {'high', 'low'})
    error('sltoolbox:invalidarg', ...
        'Invalid score option %s', op);
end

opts.scheme = 'nn';
opts = slparseprops(opts, varargin{:});

if ~ismember(opts.scheme, {'nn', 'loo'})
    error('sltoolbox:invalidarg', ...
        'Invalid scheme for classification: %s', opts.scheme);
end


%% Verify regularity of blocks

rowlims = vertcat(blocks{:, 1});
rowlims = reshape(rowlims(:, 1), [2, nrows]);
collims = vertcat(blocks{1, :});
collims = reshape(collims(:, 2), [2, ncols]);

for i = 1 : nrows
    for j = 1 : ncols
        rl = rowlims(:, i);
        cl = collims(:, j);
        if ~isequal(blocks{i, j}, [rl cl])
            error('sltoolbox:sizmismatch', ...
                'The blocks are nonregular');
        end
    end
end


%% Classify

% first-batch (on first block row)
[decinds, decscores] = procbatch(scores(1, :), n, rowlims(:, 1), collims, op, opts);

% following batches

if nrows > 1
    
    for i = 2 : nrows
        [curinds, curscores] = procbatch(scores(i, :), n, rowlims(:, i), collims, op, opts);
        
        % update
        switch op
            case 'high'
                to_replace = curscores > decscores;
            case 'low'
                to_replace = curscores < decscores;
        end
        decinds(to_replace) = curinds(to_replace);
        decscores(to_replace) = curscores(to_replace);        
    end
    
end

% convert indices to decision labels
decisions = clabels(decinds);
if size(decisions, 1) > 1
    decisions = decisions';
end


%% Internal function to process each block (batch)

function [decinds, decscores] = procblock(filename, rlim, clim, op, opts)

matsiz = [rlim(2) - rlim(1) + 1, clim(2) - clim(1) + 1];
s = slreadarray(filename);
if ~isequal(size(s), matsiz)
    error('sltoolbox:sizmismatch', ...
        'Illegal size of array in %s', filename);
end

if strcmpi(opts.scheme, 'loo')
    % select the disabled scores
    sl1 = max(rlim(1), clim(1));
    sl2 = min(rlim(2), clim(2));
    if sl1 <= sl2 % contains some elements to be disabled
        % calculate the local indices
        srs = (sl1:sl2) - (rlim(1)-1);
        scs = (sl1:sl2) - (clim(1)-1);
        sdinds = (scs - 1) * matsiz(1) + srs;        
        % disable the selected elements
        switch op
            case 'high'
                s(sdinds) = -Inf;
            case 'low'
                s(sdinds) = Inf;
        end
    end
end        

indbase = rlim(1) - 1;

switch op
    case 'high'
        [decscores, decinds] = max(s, [], 1);
    case 'low'
        [decscores, decinds] = min(s, [], 1);
end
    

if indbase ~= 0
    decinds = decinds + indbase;
end



function [decinds, decscores] = procbatch(srow, n, rlim, collims, op, opts)

decinds = zeros(1, n);
decscores = zeros(1, n);
ncols = size(srow, 2);

for j = 1 : ncols    
    sc = collims(1, j);
    ec = collims(2, j);
    [curinds, curdsco] = procblock(srow{1, j}, rlim, [sc; ec], op, opts);
    decinds(sc:ec) = curinds;
    decscores(sc:ec) = curdsco;        
end
    
