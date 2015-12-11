function [word2charMap] = loadWord2CharMap(mapFile)
%% 
% Thang Luong @ 2015 <lmthang@stanford.edu>
% Load word2charMap.
%%


  fid = fopen(mapFile,'r');
  fprintf(1, '# Loading word2char map from file %s ...\n', mapFile);
  fileLines = textscan(fid, '%s', 'delimiter', '\n');
  fclose(fid);
  fileLines=fileLines{1};

  word2charMap = cell(1,length(fileLines));
  for ii = 1:length(fileLines)
    word2charMap{ii} = strsplit(fileLines{ii},' ') + 1; %cell2mat(regexp(fileLines{ii},' ','split'));
  end
end