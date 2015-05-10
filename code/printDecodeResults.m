function printDecodeResults(decodeData, candidates, candScores, params, isOutput)
  batchSize = size(candScores, 2);
  startId = decodeData.startId;
  
  % output translations
  [maxScores, bestIndices] = max(candScores); % stackSize * batchSize
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
    fprintf(params.logId, '  score %g\n', maxScores(ii));

    % debug
    printSrc(2, decodeData, ii, params, startId+ii-1);
    printRef(2, decodeData, ii, params, startId+ii-1);
    printSent(2, translation, params.tgtVocab, ['  tgt ' num2str(startId+ii-1) ': ']);
    fprintf(2, '  score %g\n', maxScores(ii));
    %printTranslations(candidates{ii}, candScores(ii, :), params);
  end
end


function printSrc(fid, data, ii, params, sentId)
  mask = data.inputMask(ii,1:data.srcMaxLen-1);
  src = data.input(ii,mask);
  printSent(fid, src, params.srcVocab, ['  src ' num2str(sentId) ': ']);
end

function printRef(fid, data, ii, params, sentId)
  mask = data.inputMask(ii, data.srcMaxLen:end);
  ref = data.tgtOutput(ii,mask);
  printSent(fid, ref, params.tgtVocab, ['  ref ' num2str(sentId) ': ']);
end

% function printTranslations(candidates, scores, params)
%   for jj = 1 : length(candidates)
%     assert(isempty(find(candidates{jj}>params.tgtVocabSize, 1)));
%     printSent(2, candidates{jj}, params.vocab, ['cand ' num2str(jj) ', ' num2str(scores(jj)) ': ']);
%   end
% end

%     if params.separateEmb==1 
%     else
%       printSent(2, translation, params.vocab, ['  tgt ' num2str(startId+ii-1) ': ']);
%     end

%     % separate emb
%     if params.separateEmb==1 
%     else
%       printSent(params.logId, translation, params.vocab, ['  tgt ' num2str(startId+ii-1) ': ']);
%     end

%   % separate emb
%   if params.separateEmb==1 
%   else
%     printSent(fid, src, params.vocab, ['  src ' num2str(sentId) ': ']);
%   end

%   % separate emb
%   if params.separateEmb==1   
%   else
%     printSent(fid, ref, params.vocab, ['  ref ' num2str(sentId) ': ']);
%   end
