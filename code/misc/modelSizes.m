function [modelSize] = modelSizes(model)
%%%
%
% This method prints out detailed sizes of individual matrices and return
% the total model size.
% model is a struct, in which each element could be a matrix or a cell of
% matrices.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  fieldNames = fields(model);
  
  % compute model size
  modelSize = 0;
  for ii=1:length(fieldNames)
    field = fieldNames{ii};
    if iscell(model.(field))
      for jj=1:length(model.(field))
        modelSize = modelSize + numel(model.(field){jj});
        modelSizes.(field){jj} = numel(model.(field){jj});
      end
    else
      modelSize = modelSize + numel(model.(field));
      modelSizes.(field) = numel(model.(field));
    end
  end
  fprintf(2, '  Model size = %d, individual sizes: %s\n', modelSize, wInfo(modelSizes, 0));
end