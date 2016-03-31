function [We, words] = loadWeWords(modelFile, modelFormat)
% modelFormat:
% 0 -- Matlab file
% 1 -- text file with a header line <numWords> <embDim>.  Subsequent lines has <word> <values>
% 2 -- text file with each line has <word> <values>
% 3 -- assume that there are two files modelFile.We, modelFile.words

  verbose=0;
  %% load from Matlab mat file
  if modelFormat==0
    %% load We
    load(modelFile, 'We', 'words');
    if ~exist('We', 'var') 
      load(modelFile , 'allW');

      if isfield(allW, 'We') 
        We = allW.We;
      else
        error('evaluateWordSim: no We in %s', modelFile);
      end
    end

    %% load vocab
    if ~exist('words', 'var')
      error('No words in the model file %s\n', modelFile);
    end
  elseif modelFormat==3 % modelFile.We, modelFile.words
    % words
    fid = fopen([modelFile '.words'], 'r');
    tmp = textscan(fid, '%s');
    words = tmp{1};
    fclose(fid);
    We = dlmread([modelFile '.We']);
    We = We';
  %% load from a text file
  else 
    fid = fopen(modelFile, 'r');
    line = fgetl(fid); 
    assert(ischar(line)); 
    if modelFormat==1 % header line <numWord> <embDim>
      [~, embDimStr] = strtok(line, ' ');
      embDim = str2double(embDimStr);
      line = fgetl(fid);
    else
      tokens = strsplit(line);
      embDim = length(tokens)-1;
    end

    if verbose==1
      fprintf(2, '# Reading file %s, embDim=%d\n', modelFile, embDim);
    end

    words = {};
    We = [];
    numWords = 0;
    while ischar(line)
      line = strtrim(line);
      if strcmp(line, '')
        line = fgetl(fid);
        continuel
      end

      [word, valueStr] = strtok(line, ' ');
      % word
      words = [words word];
      % emb
      We = [We sscanf(valueStr, '%f')];

      numWords = numWords+1;
      if verbose==1 && (mod(numWords, 10000)==0)
        fprintf(2, ' (%d)', numWords);
      end
      line = fgetl(fid);
    end

    if verbose==1
      fprintf(2, ' Done! Num words = %d\n', numWords);
    end
  end
end
