function [data] = vec2struct(vec, sizeInfos)
%%%
%
% This method deflattens a vector of values into a structure based on the
% sizeInfos.
%
% 'sizeInfos' is a struct, in which each element could be the size info of a matrix or a cell of
% multiple size infos. 
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%%
  if ~exist('names', 'var')
    names = fields(sizeInfos);
  end
  
  % compute model size
  index = 0;
  for ii=1:length(names)
    field = names{ii};
    if iscell(sizeInfos.(field)) % cell
      for jj=1:length(sizeInfos.(field))
        matSize = sizeInfos.(field){jj};
        data.(field){jj} = reshape(vec(index+1:index+(matSize(1))*matSize(2)),matSize(1),matSize(2));
        index = index+(matSize(1))*matSize(2);
      end
    else
      matSize = sizeInfos.(field);
      data.(field) = reshape(vec(index+1:index+(matSize(1))*matSize(2)),matSize(1),matSize(2));
      index = index+(matSize(1))*matSize(2);
    end
  end
end