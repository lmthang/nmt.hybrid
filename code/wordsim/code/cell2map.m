function [map] = cell2map(cell)
%%
% Minh-Thang Luong
%
% Convert a cell (a cell) into a map by mapping cell values into their indices
%%
  if isempty(cell)
    map = containers.Map();
  else
    map = containers.Map(cell, 1:length(cell)); %num2cell(1:length(cell)));
  end
