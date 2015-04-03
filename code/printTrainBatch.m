function printTrainBatch(data, params)
  if params.separateEmb==1
    if params.isBi
      printSent(2, data.srcInput(1, :), params.srcVocab, '  src input 1:');
    end
    printSent(2, data.tgtInput(1, :), params.tgtVocab, '  tgt input 1:');
    printSent(2, data.tgtOutput(1, :), params.tgtVocab, '  tgt output 1:');
    if params.posModel>0
      printSent(2, data.posOutput(1, :), params.tgtVocab, '  srcPos 1:');
    end  
  else
    if params.isBi
      printSent(2, data.srcInput(1, :), params.vocab, '  src input 1:');
    end
    printSent(2, data.tgtInput(1, :), params.vocab, '  tgt input 1:');
    printSent(2, data.tgtOutput(1, :), params.vocab, '  tgt output 1:');
    if params.posModel>0
      printSent(2, data.posOutput(1, :), params.vocab, '  srcPos 1:');
    end
  end
  
  fprintf(2, 'src mask: %s\n', num2str(data.srcMask(1, :)));
  fprintf(2, 'tgt mask: %s\n', num2str(data.tgtMask(1, :)));
end