function [accumX, uniqIndices] = aggregateMatrix(X, indices, isGPU, dataType)
%%%
% Fast way of aggregating column vectors based on indices which contain repetitive values.
%   X = [101:105; 11:15];
%   indices = [1 4 4 2 2]
% aggregateMatrix(X, indices) returns:
%   accumX = 
%     101   209   205
%      11    29    25
%   uniqIndices = [1 2 4]
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%%%

  [uniqIndices, ~, J] = unique(indices);
  numUniqIndices = length(uniqIndices);
  numEmbGrads = length(indices);

  if numEmbGrads==1
    accumX = X;
  else
    if isGPU
      sparseMatrix = zeros(numEmbGrads, numUniqIndices, dataType, 'gpuArray');
      sparseIndices = sub2ind([numEmbGrads, numUniqIndices], 1:numEmbGrads, J'); 
      sparseMatrix(sparseIndices) = ones(numEmbGrads, 1);
    else
      sparseMatrix = sparse(1:numEmbGrads, J, ones(numEmbGrads,1), numEmbGrads, numUniqIndices);
    end
    accumX = X*sparseMatrix;
  end
end
