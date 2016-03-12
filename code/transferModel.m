function transferModel(modelFile, newSrcVocabFile, newTgtVocabFile, outModelFile)
  addpath(genpath(sprintf('%s/../..', pwd)));
  
  % load current data
  fprintf(2, '# Loading %s\n', modelFile);
  savedData = load(modelFile);
  model = savedData.model;
  params = savedData.params;
  
  % load vocabs
  [srcVocab_new] = loadVocab(newSrcVocabFile);
  [tgtVocab_new] = loadVocab(newTgtVocabFile);
  
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
  fprintf(2, '# src short list %d, num overlap %d\n', params.srcCharShortList, sum(flags));
  
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
  fprintf(2, '# tgt short list %d, num overlap %d\n', params.tgtCharShortList, sum(flags));
  
  % save model
  save(outModelFile, 'model', 'params');
end
  