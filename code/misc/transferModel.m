function transferModel(modelFile, srcVocabFile_new, tgtVocabFile_new, srcCharPrefix_new, tgtCharPrefix_new, outModelFile, varargin)
  addpath(genpath(sprintf('%s/../..', pwd)));
  
  %% Argument Parser
  p = inputParser;
  % required
  addRequired(p,'modelFile',@ischar);
  addRequired(p,'srcVocabFile_new',@ischar);
  addRequired(p,'tgtVocabFile_new',@ischar);
  addRequired(p,'srcCharPrefix_new',@ischar);
  addRequired(p,'tgtCharPrefix_new',@ischar);
  addRequired(p,'outModelFile',@ischar);
    
  % optional
  addOptional(p,'gpuDevice', 0, @isnumeric); % choose the gpuDevice to use: 0 -- no GPU 
  
  p.KeepUnmatched = true;
  parse(p, modelFile, srcVocabFile_new, tgtVocabFile_new, srcCharPrefix_new, tgtCharPrefix_new, outModelFile, varargin{:})
  transferParams = p.Results;
  
  % GPU settings
  transferParams.isGPU = 0;
  if transferParams.gpuDevice
    n = gpuDeviceCount;  
    if n>0 % GPU exists
      fprintf(2, '# %d GPUs exist. So, we will use GPUs.\n', n);
      transferParams.isGPU = 1;
      gpuDevice(transferParams.gpuDevice)
      transferParams.dataType = 'single';
    else
      transferParams.dataType = 'double';
    end
  else
    transferParams.dataType = 'double';
  end
  printParams(2, transferParams);
  
  modelFile = transferParams.modelFile;
  srcVocabFile_new = transferParams.srcVocabFile_new;
  tgtVocabFile_new = transferParams.tgtVocabFile_new;
  srcCharPrefix_new = transferParams.srcCharPrefix_new;
  tgtCharPrefix_new = transferParams.tgtCharPrefix_new;
  outModelFile = transferParams.outModelFile;
  
  %% Transfer
  % load current data
  fprintf(2, '# Loading %s\n', modelFile);
  savedData = load(modelFile);
  model = savedData.model;
  params = savedData.params;
  
  % src word
  fprintf(2, '# Transfering src word\n');
  [model.W_emb_src, params.srcVocab, params.srcVocabFile] = transferEmb(model.W_emb_src, params.srcVocab, params.srcCharShortList, ...
    srcVocabFile_new, params);
  
  % tgt word
  fprintf(2, '# Transfering tgt word\n');
  [model.W_emb_tgt, params.tgtVocab, params.tgtVocabFile] = transferEmb(model.W_emb_tgt, params.tgtVocab, params.tgtCharShortList, ...
    tgtVocabFile_new, params);
  
  % src char
  fprintf(2, '# Transfering src char\n');
  [model.W_emb_src_char, params.srcCharVocab, params.srcCharVocabFile] = transferEmb(model.W_emb_src_char, params.srcCharVocab, ...
    params.srcCharVocabSize, [srcCharPrefix_new '.char.vocab'], params);
  params.srcCharMapFile = [srcCharPrefix_new '.char.map'];
  params.srcCharMap = loadWord2CharMap(params.srcCharMapFile, params.charMaxLen);
  
  % tgt char
  fprintf(2, '# Transfering tgt char\n');
  [model.W_emb_tgt_char, params.tgtCharVocab, params.tgtCharVocabFile] = transferEmb(model.W_emb_tgt_char, params.tgtCharVocab, ...
    params.tgtCharVocabSize, [tgtCharPrefix_new '.char.vocab'], params);
  params.tgtCharMapFile = [tgtCharPrefix_new '.char.map'];
  params.tgtCharMap = loadWord2CharMap(params.tgtCharMapFile, params.charMaxLen);
  
  % save model
  save(outModelFile, 'model', 'params');
end


function [W_emb_new, vocab_new, vocabFile_new] = transferEmb(W_emb, vocab, shortList, vocabFile_new, params)
  fprintf(2, '  W_emb [%s], original vocab size %d, short list %d\n', num2str(size(W_emb)), length(vocab), shortList);
  [vocab_new] = loadVocab(vocabFile_new);
  vocabMap = cell2map(vocab(1:shortList));
  W_emb_new = initMatrixRange(params.initRange, [params.lstmSize, shortList], params.isGPU, params.dataType);
  vocabShortList_new = vocab_new(1:shortList);
  flags = isKey(vocabMap, vocabShortList_new);
  indices = values(vocabMap, vocabShortList_new(flags));
  indices = [indices{:}];
  W_emb_new(:, flags) = W_emb(:, indices);
  
  fprintf(2, '  num overlap %d\n  new words:', sum(flags));
  fprintf(2, ' %s', vocabShortList_new{~flags});
  fprintf(2, '\n');
end

%   srcVocabMap = cell2map(params.srcVocab(1:params.srcCharShortList));
%   W_emb_src_new = initMatrixRange(params.initRange, [params.lstmSize, params.srcCharShortList], params.isGPU, params.dataType);
%   srcVocabShortList_new = srcVocab_new(1:params.srcCharShortList);
%   flags = isKey(srcVocabMap, srcVocabShortList_new);
%   indices = values(srcVocabMap, srcVocabShortList_new(flags));
%   indices = [indices{:}];
%   W_emb_src_new(:, flags) = model.W_emb_src(:, indices);
%   model.W_emb_src = W_emb_src_new;
%   params.srcVocab = srcVocab_new;
%   params.srcVocabFile = srcVocabFile_new;
%   fprintf(2, '  src short list %d, num overlap %d\n  new words:', params.srcCharShortList, sum(flags));
%   fprintf(2, ' %s', srcVocabShortList_new{~flags});
%   fprintf(2, '\n');

%   tgtVocabMap = cell2map(params.tgtVocab(1:params.tgtCharShortList));
%   W_emb_tgt_new = initMatrixRange(params.initRange, [params.lstmSize, params.tgtCharShortList], params.isGPU, params.dataType);
%   tgtVocabShortList_new = tgtVocab_new(1:params.tgtCharShortList);
%   flags = isKey(tgtVocabMap, tgtVocabShortList_new);
%   indices = values(tgtVocabMap, tgtVocabShortList_new(flags));
%   indices = [indices{:}];
%   W_emb_tgt_new(:, flags) = model.W_emb_tgt(:, indices);
%   model.W_emb_tgt = W_emb_tgt_new;
%   params.tgtVocab = tgtVocab_new;
%   params.tgtVocabFile = tgtVocabFile_new;
%   fprintf(2, '  tgt short list %d, num overlap %d\n  new words:', params.tgtCharShortList, sum(flags));
%   fprintf(2, ' %s', tgtVocabShortList_new{~flags});
%   fprintf(2, '\n');
  