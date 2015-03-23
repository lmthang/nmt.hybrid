function [linearIndices] = getTensorLinearIndices(matrix, yIds, zIds)
%%%
%
% Matrix is a 3-D tensor. We want to retrieve the linear indices
%   to access the following elements:
%   [ matrix(:, yIds(1), zIds(1); ...; matrix(:, yIds(end), zIds(end)) ]
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  xDim = size(matrix, 1);
  numData = length(yIds);
  numElements = numData * xDim;
  
  %xIds = repmat(1:xDim, 1, numData);
  xIds = kron(ones(1, numData), 1:xDim);
  yIds = reshape(repmat(yIds, xDim, 1), 1, numElements);
  zIds = reshape(repmat(zIds, xDim, 1), 1, numElements);
  linearIndices = sub2ind(size(matrix), xIds, yIds, zIds);
  
%   if params.assert
%     assert(length(linearIndices)==length(unique(linearIndices)));
%   end
end