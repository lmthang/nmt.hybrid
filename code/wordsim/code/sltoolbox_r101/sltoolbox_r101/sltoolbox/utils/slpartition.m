function PS = slpartition(whole_size, spec_item, varargin)
%SLPARTITION Partition a range into blocks in a specified manner
%
% $ Syntax $
%   - PS = slpartition(whole_size, 'numblks', nblks_dim1, nblks_dim2, ...);
%   - PS = slpartition(whole_size, 'numblks', [nblks_dim1, nblks_dim2, ...]);
%   - PS = slpartition(whole_size, 'maxblksize', mbs_dim1, mbs_dim2, ...);
%   - PS = slpartition(whole_size, 'maxblksize', [mbs_dim1, mbs_dim2, ...]);
%   - PS = slpartition(whole_size, 'blksizes', blksizes_dim1, blksizes_dim2, ...);
%   - PS = slpartition(whole_size, 'startinds', startinds_dim1, startinds_dim2, ...);
%   - PS = slpartition(whole_size, 'endinds', endinds_dim1, endinds_dim2, ...);
%
% $ Description $
%   
%   - PS = slpartition(whole_size, 'numblks', nblks_dim1, nblks_dim2, ...) 
%     PS = slpartition(whole_size, 'numblks', [nblks_dim1, nblks_dim2, ...])
%     partitions the single-dimensional or multi-dimensional range with its
%     size specified in row vector whole_size, into multiple blocks, and
%     outputs the partition structure via PS. PS is a struct-array, with
%     d entries, where d is the dimension of the whole array. Each entry
%     of PS is a struct with two fields: sinds and einds, which are
%     row vectors respectively denoting the start indices and end indices
%     of sequential blocks along the dimension.
%     By this syntax, the user can specify the number of blocks along each
%     dimension by nblks_dim1, nblks_dim2, .... They are all integers.
%
%   - PS = slpartition(whole_size, 'maxblksize', mbs_dim1, mbs_dim2, ...)
%     PS = slpartition(whole_size, 'maxblksize', [mbs_dim1, mbs_dim2, ...])
%     By this syntax, the user can specify the maximum block sizes along
%     each dimension by mbs_dim1, mbs_dim2, .... They are all integers.
%
%   - PS = slpartition(whole_size, 'blksizes', blksizes_dim1, blksizes_dim2, ...)
%     By this syntax, the user can specify the block sizes of all individual
%     blocks in all dimensions. blksizes_dim1, blksizes_dim2, ...
%     are integer vectors, with each entry specifying the length of a block
%     in along some dimension.
%
%   - PS = slpartition(whole_size, 'startinds', startinds_dim1, startinds_dim2, ...);
%     PS = slpartition(whole_size, 'endinds', endinds_dim1, endinds_dim2, ...);
%     By this syntax, the user can specify the start indices or end indices
%     of the blocks in all dimensions. startinds_dim1, startinds_dim2, ...
%     are integer vectors, with entry specifying the start index of a block
%     along a certain dimension. Likewise, endinds_dim1, endinds_dim2, ...
%     are also integer vectors, with entry specifying the ending index of 
%     a block along a certain dimension.
%
% $ Remarks $
%   - For the dimension, corresponding parameter is not given, the dimension
%     is considered to be partitioned into multiple unit blocks with
%     block length equaling 1 along that dimension.
%
% $ History $
%   - Created by Dahua Lin on Dec 7th, 2005
%   - Modified by Dahua Lin on Sep 10, 2006
%       - make minor change to eliminate warnings
%

%% parse and verify input arguments
if nargin < 3
    raise_lackinput('slpartition', 3);
end
dim_whole = length(whole_size);
if ~isequal(size(whole_size), [1, dim_whole])
    error('sltoolbox:notsizevec', ...
        'The size vector whole_size should be a row vector');
end
if any(whole_size <= 0)
    error('sltoolbox:emptyarray', ...
        'The whole_size corresponds to an empty array');
end
% compute the actual dimension
dim_whole = find(whole_size > 1, 1, 'last');
whole_size = whole_size(1:dim_whole);

%% Compute

% initialize the struct
PS = struct('sinds', cell(dim_whole, 1), 'einds', []);

% process
switch spec_item
    case 'numblks'
        
        if numel(varargin{1}) > 1
            G = varargin{1};
        else
            G = cell2mat(varargin);
        end
        nG = length(G);
        
        for d = 1 : dim_whole            
            if d <= nG
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_numblks(whole_size(d), G(d));
            else
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_full(whole_size(d));
            end
        end        
        
    case 'maxblksize'
        
        if numel(varargin{1}) > 1
            G = varargin{1};
        else
            G = cell2mat(varargin);
        end
        nG = length(G);
        
        for d = 1 : dim_whole            
            if d <= nG
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_maxblksize(whole_size(d), G(d));
            else
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_full(whole_size(d));
            end
        end     
        
    case 'blksizes'
        
        G = varargin;
        nG = length(G);
        
        for d = 1 : dim_whole            
            if d <= nG
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_blksizes(whole_size(d), G{d});
            else
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_full(whole_size(d));
            end
        end             
        
    case 'startinds'
        
        G = varargin;
        nG = length(G);
        
        for d = 1 : dim_whole            
            if d <= nG
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_startinds(whole_size(d), G{d});
            else
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_full(whole_size(d));
            end
        end         
        
    case 'endinds'
        
        G = varargin;
        nG = length(G);
        
        for d = 1 : dim_whole            
            if d <= nG
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_endinds(whole_size(d), G{d});
            else
                [PS(d).sinds, PS(d).einds] = ...
                    partition1d_full(whole_size(d));
            end
        end 
        
    otherwise
        error('sltoolbox:invalid_item', ...
            'Invalid partition item name: %s', spec_item);
end


%% ============ single-dimension partition functions =============

function check_validity(siz, s, e)

if length(s) ~= length(e)
    error('sltoolbox:invalid_partition', ...
        'The number of start indices and that of end indices do not consist');
end
if any(s > e)
    error('sltoolbox:invalid_partition', ...
        'Found some starting indices is larger than corresponding end indices');
end
if ~isequal(e(1:end-1)+1, s(2:end))
    error('sltoolbox:invalid_partition', ...
        'The regions are not contingent');
end
if s(1) ~= 1
    error('sltoolbox:invalid_partition', ...
        'The first starting index is not equal to 1');
end
if e(end) ~= siz
    error('sltoolbox:invalid_partition', ...
        'The last end index is not equal to dimension');
end


function [s, e] = partition1d_full(siz)

s = 1:siz;
e = s;
check_validity(siz, s, e);


function [s, e] = partition1d_numblks(siz, nblks)

if nblks > siz
    error('The number of blocks should not exceed the whole dimension');
end
b = round(linspace(1, siz+1, nblks+1));
s = b(1:nblks);
e = b(2:nblks+1)-1;
check_validity(siz, s, e);


function [s, e] = partition1d_maxblksize(siz, mbs)

nblks = ceil(siz / mbs);
if nblks == 1
    blksizes = siz;
else
    blksizes = [mbs * ones(1, nblks-1), siz - mbs * (nblks-1)];
end
[s, e] = partition1d_blksizes(siz, blksizes);


function [s, e] = partition1d_blksizes(siz, blksizes)

if sum(blksizes) ~= siz
    error('The total block sizes is not equal to the whole size');
end
if any(blksizes <= 0)
    error('Some block sizes are less than or equal to zero');
end
e = cumsum(blksizes);
s = [1, e(1:end-1)+1];
check_validity(siz, s, e);


function [s, e] = partition1d_startinds(siz, startinds)

if startinds(1) ~= 1
    error('The first starting index should be equal to 1');
end
if startinds(end) > siz
    error('The last starting index should not exceeds the whole size');
end
if any(diff(startinds) <= 0)
    error('The order of starting indices is incorrect');
end

s = startinds;
e = [startinds(2:end)-1, siz];
check_validity(siz, s, e);


function [s, e] = partition1d_endinds(siz, endinds)

if endinds(1) < 1
    error('The first ending index should be not less than 1');
end
if endinds(end) ~= siz
    error('The last ending index should be equal to the whole size');
end
if any(diff(endinds) <= 0)
    error('The order of ending indices is incorrect');
end

e = endinds;
s = [1, e(1:end-1)+1];
check_validity(siz, s, e);
    
















