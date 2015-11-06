function printDecodeResults(decodeData, candidates, candScores, alignInfo, params, isOutput)
  batchSize = size(candScores, 2);
  startId = decodeData.startId;
  
  % output translations
  [maxScores, bestIndices] = max(candScores, [], 1); % stackSize * batchSize
  for ii = 1:batchSize
    bestId = bestIndices(ii);
    translation = candidates{ii}{bestId}; 
    assert(isempty(find(translation>params.tgtVocabSize, 1)));
    
    if isOutput
      printSent(params.fid, translation(1:end-1), params.tgtVocab, ''); % remove <t_eos>
    end

    % log
    printSrc(params.logId, decodeData, ii, params, startId+ii-1);
    printRef(params.logId, decodeData, ii, params, startId+ii-1);
    printSent(params.logId, translation, params.tgtVocab, ['  tgt ' num2str(startId+ii-1) ': ']);    
    % align
    if params.align
      alignment = alignInfo{ii}{bestId};
      printAlign(params.logId, translation, decodeData, alignment, params, ii, startId+ii-1, 1);
      if isOutput
        printSentAlign(params.alignId, translation, decodeData, alignment, ii, params);
      end
    end
    fprintf(params.logId, '  score %g\n', maxScores(ii));

    % debug
    printSrc(2, decodeData, ii, params, startId+ii-1);
    printRef(2, decodeData, ii, params, startId+ii-1);
    printSent(2, translation, params.tgtVocab, ['  tgt ' num2str(startId+ii-1) ': ']);
    % align
    if params.align
      printAlign(2, translation, decodeData, alignment, params, ii, startId+ii-1, 1);
    end
    fprintf(2, '  score %g\n', maxScores(ii));
  end
end

function printSentAlign(fid, translation, data, alignment, ii, params)
  srcLen = data.srcLens(ii);
  if params.isReverse
    alignment = srcLen-alignment;
  end

  tgtLen = length(translation);
  for i = 1:(tgtLen-1)
    srcPos = alignment(i+1);
    assert((1 <= srcPos) && (srcPos <= srcLen), sprintf('Assertion failed: srcPos = %d, srcLen = %d', srcPos, srcLen));
    fprintf(fid, '%d-%d ', srcPos-1, i-1); % base 0
  end
  fprintf(fid, '\n');
end

function printAlign(fid, translation, data, alignment, params, ii, sentId, printWords)
  mask = data.inputMask(ii,1:data.srcMaxLen-1);
  srcLen = data.srcLens(ii);

  if params.isReverse
    alignment = srcLen-alignment;
  end
  
  tgtLen = length(translation);
  if printWords
    src = data.input(ii,mask);
    if params.isReverse
      src = src(end:-1:1);
    end
    fprintf(fid, 'pairs %d: ', sentId);
    for i = 1:(tgtLen-1)
      srcPos = alignment(i+1);
      assert((1 <= srcPos) && (srcPos <= srcLen), sprintf('Assertion failed: srcPos = %d, srcLen = %d', srcPos, srcLen));
      fprintf(fid, '%s-%s ', params.srcVocab{src(srcPos)}, params.tgtVocab{translation(i)});
    end
    fprintf(fid, '\n');
  end  
  
  fprintf(fid, 'align %d: ', sentId);
  for i = 1:(tgtLen-1)
    srcPos = alignment(i+1);
    assert((1 <= srcPos) && (srcPos <= srcLen));
    fprintf(fid, '%d-%d ', srcPos-1, i-1); % base 0
  end
  fprintf(fid, '\n');
  
end


function printSrc(fid, data, ii, params, sentId)
  mask = data.inputMask(ii,1:data.srcMaxLen-1);
  src = data.input(ii,mask);
  printSent(fid, src, params.srcVocab, ['  src ' num2str(sentId) ': ']);
  
  if params.isReverse
    printSent(fid, src(end:-1:1), params.srcVocab, [' rsrc ' num2str(sentId) ': ']);
  end
end

function printRef(fid, data, ii, params, sentId)
  mask = data.inputMask(ii, data.srcMaxLen:end);
  ref = data.tgtOutput(ii,mask);
  printSent(fid, ref, params.tgtVocab, ['  ref ' num2str(sentId) ': ']);
end