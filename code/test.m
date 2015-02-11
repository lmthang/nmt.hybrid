function [] = test()
  addpath(genpath(sprintf('%s/../../matlab', pwd)));
  addpath(genpath(sprintf('%s/..', pwd)));
  
  load('/scr/nlp/deeplearning/lmthang/lstm/lstm.deen.50000.d1000.lr1.max5.d2.init0.1.noClip/modelRecent.mat');
  % params.isReverse = 0;
  % params.validPrefix = '../output/valid';
  [srcVocab] = params.vocab(params.tgtVocabSize+1:end);
  [tgtVocab] = params.vocab(1 : params.tgtVocabSize);
  validData  = loadPrepareData(params, params.validPrefix, srcVocab, tgtVocab);
  
  [candidates, scores] = lstmDecoder(model, validData.input, validData.inputMask, validData.srcMaxLen, params, 3);
  beamSize = 3;
  for i = 1:length(candidates)
    mask = find(validData.inputMask(i,1:validData.srcMaxLen));
    src = validData.input(i,mask(:));
    printSent(src, params.vocab, 'source: ');
    if isempty(candidates{i})
      fprintf(2, 'no translations.\n');
    else
      for j = 1 : length(candidates{i})
        printSent(candidates{i}{j}, params.vocab, 'candidate: ');
      end
    end
    fprintf(2, '=======================================\n');
  end
end

function [data] = loadPrepareData(params, prefix, srcVocab, tgtVocab)
  % src
  if params.isBi
    if params.isReverse
      srcFile = sprintf('%s.reversed.%s', prefix, params.srcLang);
    else
      srcFile = sprintf('%s.%s', prefix, params.srcLang);
    end
    [srcSents] = loadMonoData(srcFile, params.srcEos, -1, params.baseIndex, srcVocab, 'src');
  else
    srcSents = {};
  end
  
  % tgt
  tgtFile = sprintf('%s.%s', prefix, params.tgtLang);
  [tgtSents] = loadMonoData(tgtFile, params.tgtEos, -1, params.baseIndex, tgtVocab, 'tgt');

  % prepare
  [data.input, data.inputMask, data.tgtOutput, data.srcMaxLen, data.tgtMaxLen, data.numWords] = prepareData(srcSents, tgtSents, params);
  
  fprintf(2, '  numWords=%d\n', data.numWords);
end

function [sents, numSents] = loadMonoData(file, eos, numSents, baseIndex, vocab, label)
  fprintf(2, '# Loading data %s from file %s\n', label, file);
  fid = fopen(file, 'r');
  [sents, numSents] = loadBatchData(fid, baseIndex, numSents, eos);
  fclose(fid);
%   printSent(sents{1}, vocab, ['  ', label, ' 1:']);
%   printSent(sents{end}, vocab, ['  ', label, ' end:']);
end

function [sents, numSents] = loadBatchData(fid, baseIndex, batchSize, suffix)
  if ~exist('suffix', 'var')
    suffix = [];
  end
  
  sents = cell(batchSize, 1);
  numSents = 0;
  while ~feof(fid)
    indices = sscanf(fgetl(fid), '%d') + (1-baseIndex);
    if isempty(indices) % ignore empty lines
      continue
    end
    
    numSents = numSents + 1;
    sents{numSents} = [indices' suffix];
    if numSents==batchSize
      break;
    end
  end
  sents((numSents+1):end) = []; % delete empty cells
end
% 
% 
% function printSent(sent, vocab, prefix)
%   fprintf(2, '%s', prefix);
%   for ii=1:length(sent)
%     fprintf(2, ' %s', vocab{sent(ii)}); 
%   end
%   fprintf(2, '\n');
% end
