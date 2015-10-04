function [vec, sizeInfos] = struct2vec(data, names)
%%%
%
% This method flattens out a structure into a vector of values.
%
% 'data' is a struct, in which each element could be a matrix or a cell of
% matrices. 
% 'names' can be specified to limit what matrices to flatten.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  if ~exist('names', 'var')
    names = fields(data);
  end
  
  % compute model size
  vec = [];
  for ii=1:length(names)
    field = names{ii};
    if iscell(data.(field)) % cell
      for jj=1:length(data.(field))
        vec = [vec ; data.(field){jj}(:)];
        sizeInfos.(field){jj} = size(data.(field){jj});
      end
    else
      vec = [vec ; data.(field)(:)];
      sizeInfos.(field) = size(data.(field));
    end
  end
end