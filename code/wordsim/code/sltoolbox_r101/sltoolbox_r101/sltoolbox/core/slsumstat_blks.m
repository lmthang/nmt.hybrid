function R = slsumstat_blks(data, statfunc, varargin)
%SLSUMSTAT_BLKS Sums up statistics on all blocks for partitioned data
%
% $ Syntax $
%   - R = slsumstat_blks(data, statfunc, ...)
%
% $ Arguments $
%   - data:         the data array of cell array of data array filenames
%   - statfunc:     the function to compute statistics on data array
%   - R:            the sum statistics
%
% $ Description $
%   - R = slsumstat_blks(data, statfunc, ...) computes statistics on
%     all data blocks and sums them by plus. When data is an array,
%     it just invokes statfunc on it and return, if data is a cell array
%     of filenames, it loads data of each block, computes statistics
%     blockwise and sums them up to give the final result.
%
% $ Remarks $
%   - The function is widely applicable for diverse types of computation.
%     The only conditions is that the function values on the whole matrix
%     is equivalent to the sum of function values on all blocks.
% 
%   - The function values can be either a scalar or an array of any 
%     dimensions, provided that the values produced on all blocks have
%     equal size.
%
% $ History $
%   - Created by Dahua Lin, on Aug 8th, 2006
%

%% parse and verify input arguments

if nargin < 2
    raise_lackinput('slsumstat_blks', 2);
end
if ~isnumeric(data) && ~iscell(data)
    error('sltoolbox:invalidargs', ...
        'data should be either a numeric array or a cell array of strings');
end


%% compute

if isnumeric(data)
    R = feval(statfunc, data, varargin{:});
else
    n = numel(data);
    
    % first section
    curdata = slreadarray(data{1});
    R = feval(statfunc, curdata, varargin{:});
    
    % other section
    if n > 1
        for i = 2 : n
            curdata = slreadarray(data{i});
            curR = feval(statfunc, curdata, varargin{:});
            R = R + curR;
        end
    end
end



