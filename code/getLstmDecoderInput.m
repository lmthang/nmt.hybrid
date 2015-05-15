
function [x_t, inputInfo] = getLstmDecoderInput(decodeInput, tgtPos, W_emb, softmax_h, trainData, zeroState, params) %, curMask)
  inputInfo = [];
  
  % same-length decoder
  if params.sameLength==1
    if tgtPos>params.numSrcHidVecs
      x_t = [W_emb(:, decodeInput); zeroState];
    else
      x_t = [W_emb(:, decodeInput); trainData.srcHidVecs(:, :, params.numSrcHidVecs-tgtPos+1)];
    end
  
  % feed softmax vector of the previous timestep
  elseif params.softmaxFeedInput
    x_t = [W_emb(:, decodeInput); softmax_h];
    
%   % positionl models 2: at the first level, we use additional src information
%   elseif params.posModel==2 && mod(tgtPos, 2)==0 % predict words
%     positions = decodeInput;
%     
%     [s_t, inputInfo.srcPosLinearIndices] = buildSrcPosVecs(tgtPos, params, trainData, positions, curMask);
%     x_t = [W_emb(:, decodeInput); s_t];
  else
    x_t = W_emb(:, decodeInput);
  end
end