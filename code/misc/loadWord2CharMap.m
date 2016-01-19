function [word2charMap] = loadWord2CharMap(mapFile)
%% 
% Thang Luong @ 2015 <lmthang@stanford.edu>
% Load word2charMap.
%%


  fprintf(1, '# Loading word2char map from file %s ...\n', mapFile);
  fid = fopen(mapFile,'r');
  word2charMap = cell(1, 1000000);
  ii = 1;
  while ~feof(fid)
    word2charMap{ii} = sscanf(fgetl(fid), '%d') + 1;
    ii = ii + 1;
  end
  word2charMap(ii:end) = [];
end