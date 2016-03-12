function transferModel(modelFile, srcVocabFile_new, tgtVocabFile_new, outModelFile)
  addpath(genpath(sprintf('%s/../..', pwd)));
  
  % load current data
  fprintf(2, '# Loading %s\n', modelFile);
  savedData = load(modelFile);
  model = savedData.model;
  params = savedData.params;
  
  % load vocabs
  [srcVocab_new] = loadVocab(srcVocabFile_new);
  [tgtVocab_new] = loadVocab(tgtVocabFile_new);
  
  % prepare new src model
  fprintf(2, '# Transfering src\n');
  srcVocabMap = cell2map(params.srcVocab(1:params.srcCharShortList));
  W_emb_src_new = initMatrixRange(params.initRange, [params.lstmSize, params.srcCharShortList], params.isGPU, params.dataType);
  srcVocabShortList_new = srcVocab_new(1:params.srcCharShortList);
  flags = isKey(srcVocabMap, srcVocabShortList_new);
  indices = values(srcVocabMap, srcVocabShortList_new(flags));
  indices = [indices{:}];
  W_emb_src_new(:, flags) = model.W_emb_src(:, indices);
  model.W_emb_src = W_emb_src_new;
  params.srcVocab = srcVocab_new;
  params.srcVocabFile = srcVocabFile_new;
  fprintf(2, '# src short list %d, num overlap %d\n, new words:', params.srcCharShortList, sum(flags));
  fprintf(2, ' %s', srcVocabShortList_new{~flags});
  fprintf(2, '\n');
  
  % prepare new tgt model
  fprintf(2, '# Transfering tgt\n');
  tgtVocabMap = cell2map(params.tgtVocab(1:params.tgtCharShortList));
  W_emb_tgt_new = initMatrixRange(params.initRange, [params.lstmSize, params.tgtCharShortList], params.isGPU, params.dataType);
  tgtVocabShortList_new = tgtVocab_new(1:params.tgtCharShortList);
  flags = isKey(tgtVocabMap, tgtVocabShortList_new);
  indices = values(tgtVocabMap, tgtVocabShortList_new(flags));
  indices = [indices{:}];
  W_emb_tgt_new(:, flags) = model.W_emb_tgt(:, indices);
  model.W_emb_tgt = W_emb_tgt_new;
  params.tgtVocab = tgtVocab_new;
  params.tgtVocabFile = tgtVocabFile_new;
  fprintf(2, '  tgt short list %d, num overlap %d\n  new words:', params.tgtCharShortList, sum(flags));
  fprintf(2, ' %s', tgtVocabShortList_new{~flags});
  fprintf(2, '\n');
  
  % save model
  save(outModelFile, 'model', 'params');
end
  