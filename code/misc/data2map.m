function [map] = data2map(data)
%
%  Convert a data item (a cell or a vec) into a map by mapping values into their indices
%  Thang Luong @ 2013, 2015 <lmthang@stanford.edu>
%
  if isempty(data)
    map = containers.Map();
  else
    if iscell(data)
      map = containers.Map(data, 1:length(data));
    else
      map = containers.Map(num2cell(data), 1:length(data));
    end
  end
