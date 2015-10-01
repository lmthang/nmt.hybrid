function [gradNorm, indNorms] = computeGradNorm(grad, batchSize, names)
%%%
%
% This method prints out detailed sizes of individual matrices and return
% the total model size.
%
% 'grad' is a struct, in which each element could be a matrix or a cell of
% matrices. 
% 'names' can be specified to limit what matrices to compute the
% norm.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  if ~exist('names', 'var')
    names = fields(grad);
  end
  
  % compute model size
  gradNorm = 0;
  for ii=1:length(names)
    field = names{ii};
    if iscell(grad.(field)) % cell
      for jj=1:length(grad.(field))
        indNorms.(field){jj} = double(sum(grad.(field){jj}(:).^2));
        gradNorm = gradNorm + indNorms.(field){jj};
        indNorms.(field){jj} = sqrt(indNorms.(field){jj})/batchSize;
      end
    else
      indNorms.(field) = double(sum(grad.(field)(:).^2));
      gradNorm = gradNorm + indNorms.(field);
      indNorms.(field) = sqrt(indNorms.(field))/batchSize;
    end
  end
  
  gradNorm = sqrt(gradNorm)/batchSize;
end
