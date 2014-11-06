function [stack, decodeInfo] = param2stack(varargin)

stack = [];
numArgs = length(varargin);
decodeInfo = cell(1, numArgs);
for i=1:numArgs
  if iscell(varargin{i})
      decodeCell = {}; % Thang fix bug: reset decodeCell in case varagin contains cells of different lengths
      for c = 1:length(varargin{i})
          decodeCell{c} = size(varargin{i}{c});
          stack = [stack ; varargin{i}{c}(:)];
      end
      decodeInfo{i} = decodeCell;
  else
      decodeInfo{i} = size(varargin{i});
      stack = [stack; varargin{i}(:)];
  end
end

