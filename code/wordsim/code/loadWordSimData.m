%% Load word similarity file and save in Matlab format
%% Assumed format: word1\tword2\tscore[Optional]
function [wordPairs, humanScores] = loadWordSimData(inFile, isHeader, delimiter, numLastLineExcluded, word1Index, word2Index, scoreIndex)
% outFile, 
  fid = fopen(inFile, 'r');
  fileLines = textscan(fid, '%s', 'delimiter', '\n', 'bufsize', 100000);
  fclose(fid);
  
  if ~exist('numLastLineExcluded', 'var')
    numLastLineExcluded = 0;
  end
  if ~exist('word1Index', 'var') || ~exist('word2Index', 'var') || ~exist('scoreIndex', 'var')
    word1Index = 1;
    word2Index = 2;
    scoreIndex = 3;
  end
  
  if isHeader % remove header
    startId = 2;
  else % no header
    startId = 1;
  end
  fileLines=fileLines{1}(startId:(end-numLastLineExcluded)); 
  
  wordPairs = cell(length(fileLines), 2);
  humanScores = zeros(length(fileLines),1);

  for ii = 1:length(fileLines)
    line = regexp(fileLines{ii}, delimiter, 'split');
    line = cellfun(@strtrim,line,'UniformOutput',0);
    wordPairs(ii,:) = line([word1Index word2Index]);
    humanScores(ii) = str2double(line{scoreIndex});
  end
  %save(outFile, 'wordPairs', 'humanScores');
end
