function [corrScores, data] = evaluateWordSim(modelFile, modelFormat, lang, We, words)
%%
% Run word similarity evaluation
% 
% modelFile contains either 'We', 'words' or allW, 'params', 'words'
% modelFormat: 0 -- mat file, 
%              1 -- text file with a header line <numWords> <embDim>.
%              Subsequent lines has <word> <values>
%              2 -- text file with each line has <word> <values>
% testFiles: a cell of different word similarity data files
%
% Author: Thang Luong
%%
  dataDir = '../data';
  if strcmp(lang, 'en') == 1
    dataSets = {'ws353', 'MC', 'RG', 'scws', 'rare'};
  elseif strcmp(lang, 'zh') == 1
    dataSets = {'zh'};
  elseif strcmp(lang, 'de') == 1
    dataSets = {'de'};
  elseif strcmp(lang, 'rare') == 1
    dataSets = {'rare'};
  end
  
  addpath(genpath('./sltoolbox_r101/'));
  verbose=0;
  if ~exist('We', 'var') || ~exist('words', 'var')
    [We, words] = loadWeWords(modelFile, modelFormat);
  end
  vocabMap = cell2map(words); % map words to to indices

  % turn into lowercase, needed for languages like German where the word Produktion exists only in this form.
  for ii=1:length(words)
    lowerWord = lower(words{ii});
    if ~isKey(vocabMap, lowerWord)
      vocabMap(lowerWord) = ii;
    end
  end

  %% unkStr
  unkStr = findUnkStr(vocabMap);
  if strcmp(unkStr, '')
    unkStr = '</s>';
  end

  %% Evaluation
  [corrScores, data] = simEval(We, vocabMap, unkStr, dataDir, dataSets);
end

function [corrScores, data] = simEval(We, vocabMap, unkStr, dataDir, dataSets)
  % settings
  distType = 'corrdist';
  
  numDatasets = length(dataSets);
  data = cell(1, numDatasets);
  corrScores = zeros(1, numDatasets); 
  for kk = 1:numDatasets
    testFile = [dataDir '/' dataSets{kk} '.txt'];
    
    %% read and convert vocab data
    [datum.wordPairs, datum.humanScores] = loadWordSimData(testFile, 0, '\t'); % no header, '\s+'

    simScores = getSimScores(datum.wordPairs, We, vocabMap, distType, unkStr);
    simScores(1) = simScores(1) + 1e-10; % hack to avoid N/A value return by corr() if simScores all have the same value
    corrScores(kk) = corr(simScores, datum.humanScores, 'type', 'spearman');

    data{kk} = struct('wordPairs', {datum.wordPairs}, 'humanScores', datum.humanScores, 'simScores', simScores, 'testFile', testFile);
    fprintf(2, ' %s %2.2f', dataSets{kk}, corrScores(kk)*100);
  end
  fprintf(2, '\neval wordsim'); 
  for kk = 1:numDatasets
    fprintf(2, ' %2.2f', corrScores(kk)*100);
  end
  fprintf(2, '\n'); 
end



    % find unk, exclude
    %newWordPairs = {};
    %newHumanScores = [];
    %count = 0;
    %unk_count = 0;
    %for i = 1:length(datum.wordPairs)
    %  word1 = datum.wordPairs{i,1};
    %  word2 = datum.wordPairs{i,2};

    %  % try to convert to init cap, e.g. for German, produktion is not in our vocab but Produktion is.
    %  if ~isKey(vocabMap, word1)
    %    word1(1) = upper(word1(1));
    %  end
    %  if ~isKey(vocabMap, word2)
    %    word2(1) = upper(word2(1));
    %  end
    %  if isKey(vocabMap, word1) && isKey(vocabMap, word2)
    %    count = count + 1;
    %    newWordPairs{count, 1} = word1;
    %    newWordPairs{count, 2} = word2;
    %    newHumanScores(end+1, 1) = datum.humanScores(i);
    %  else
    %    unk_count = unk_count + 1;
    %    fprintf(2, 'oov: %s\t%s\n', word1, word2);
    %  end
    %end
    %fprintf(2, '# num unks = %d\n', unk_count);
    %datum.wordPairs = newWordPairs;
    %datum.humanScores = newHumanScores;
    
