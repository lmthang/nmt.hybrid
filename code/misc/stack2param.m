function varargout = stack2param(X, decodeInfo)

%assert(length(decodeInfo)==nargout,'this should output as many variables as you gave to get X with param2stack!')

index=0;
numArgs = length(decodeInfo);
varargout = cell(1, numArgs);
for i=1:numArgs
  if iscell(decodeInfo{i})
    cellOut = cell(length(decodeInfo{i}), 1); % Thang fix
    for c = 1:length(decodeInfo{i})
      matSize = decodeInfo{i}{c};
      cellOut{c} = reshape(X(index+1:index+(matSize(1))*matSize(2)),matSize(1),matSize(2));
      index = index+(matSize(1))*matSize(2);
    end
    varargout{i}=cellOut;
  else
    matSize = decodeInfo{i};
    varargout{i} = reshape(X(index+1:index+(matSize(1))*matSize(2)),matSize(1),matSize(2));
    index = index+(matSize(1))*matSize(2);
  end
end

%   matSize = decodeInfo{i};
%   varargout{i} = reshape(X(index+1:index+(matSize(1))*matSize(2)),matSize(1),matSize(2));
%   index = index+(matSize(1))*matSize(2);



