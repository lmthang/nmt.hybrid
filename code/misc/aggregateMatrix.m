function [accumX] = aggregateMatrix(X, indices, numCols, isGPU)
%%%
% Fast way of aggregating column vectors based on indices which contain repetitive values.
%   X = [101:105; 11:15];
%   indices = [1 2 2 4 4]
% max(indices) <= numCols
%   aggregateMatrix(X, indices) returns:
%   101   205   0   209
%    11    25   0    29
%
% Thang Luong @ 2012, <lmthang@stanford.edu>
%%%

  % assert(max(indices) <= numCols);
  numIndices = length(indices);  
  
  % option1
  accumX = sparse(X)*sparse(1:numIndices, double(indices), ones(numIndices,1),numIndices, numCols);
%  if isGPU
%    dim = size(X, 1); % num dim
%    if size(indices, 1)==1 % row vector
%      indices = indices'
%    end
%    subs = [repmat((1:dim)', numel(indices), 1) kron(indices, ones(dim, 1))];
%    accumX = accumarray(subs, double(X(:)), [dim numCols], [], [], true);
%  else
%  end
end

%% I've tried tons of other ways (see below), but the current version is the fastest one %%

  % option2
  %zeroOneMatrix = sparse(1:numIndices, indices, ones(numIndices,1),numIndices, numCols);
  %accumX = sparse(X*zeroOneMatrix);
  
  % option3
  %[accumX] = aggregateColVectors(X, indices, numCols);
  
  % option4
  %[accumX] = transpose(aggregateRowVectors(X', indices));
  
% function [accumX] = aggregateColVectors(X, indices, numCols)
%   [numIndices] = size(X, 2);
%   zeroOneMatrix = sparse(1:numIndices, indices, ones(numIndices,1),numIndices, numCols);
%   accumX = X*zeroOneMatrix;
% end


% function [accumX] = aggregateRowVectors(X, indices)
%   % http://stackoverflow.com/questions/4350735/is-there-an-accumarray-that-takes-matrix-as-val
%   b = size(X, 2);
%   indices = [repmat(indices(:),b,1) ...             %# Replicate the row indices
%           kron(1:b,ones(1,numel(indices))).'];  %'# Create column indices
%   accumX = accumarray(indices,X(:));  %# I used "totals" instead of "means"
% end

%   if isCheckGrad
%     % consider a toy example
%     % indices = [101 102 102 103]
%     % X = [1 3 5 7; 2 4 6 8]
%     % if we flatten out X, by doing X(:), we'll have 1 2 3 4 5 6 7 8
%     % we need to create a flat indices that correctly accumulates X(:)
%     % let's do [uniqueIndices I J] = unique(indices);
%     % we'll have J = [1 2 2 3]
%     % we want to replace each value i in J by a row vector (i-1)*numRows+1, (i-1)*numRows+2, ..., (i-1)*numRows+numRows
%     % we accomplish that in two steps:
%     %   (1) J-1, then take kronecker product with [numRows .... numRows]
%     %   (2) then sum with a kronecker product of [1 .. 1], [1 2 ... numRows]
%     [uniqueIndices I J] = unique(indices);
%     numUniqueIndices = length(uniqueIndices);
% 
%     numRows = size(X, 1);
%     flatIndices = kron(J-ones(numIndices, 1), numRows*ones(numRows, 1)) + kron(ones(numIndices, 1), (1:numRows)');
%     newAccumX = sparse(numRows, numCols);
%     newAccumX(:, uniqueIndices) = reshape(accumarray(flatIndices, X(:)), numRows, numUniqueIndices);
% 
%     % this non-sparse version is slow %newAccumX = X*zeroOneMatrix;  
%     assert(sum(sum(abs(accumX-newAccumX)))<1e-10, '! Difference in aggregating matrices\n');
%   end
  
%   [accumX] = aggregateColVectors(X, indices, numCols);
%   tStart = tic; % measure time
%   tElapsed1 = toc(tStart);
%   tStart = tic; % measure time
%   [accumX1] = transpose(aggregateRowVectors(X', indices));
%   tElapsed2 = toc(tStart);
%   assert(sum(sum(abs(accumX-accumX1))) == 0);
%   fprintf(1, '# time = %f vs. %f\n', tElapsed1, tElapsed2);

%% results in a vector (like accumX(:))
%   numRows = size(X, 1);
%   indices(end+1) = numCols;
%   X = [X zeros(numRows, 1)];
%   
%   numIndices = length(indices);
%   subs = kron(indices-1, numRows*ones(numRows, 1)) + repmat((1:numRows)', numIndices, 1);
%   accumX = accumarray(subs, X(:));
  
