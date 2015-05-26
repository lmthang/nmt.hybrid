function [softmax_h, h2sInfo] = hid2softLayerForward(h_t, params, model, trainData, curMask, tgtPos)
%%%
%
% From lstm hidden state to softmax hidden state.
%
% Thang Luong @ 2015, <lmthang@stanford.edu>
%
%%% 
  h2sInfo = [];
  
  if params.softmaxDim || params.attnFunc %|| (params.posModel==3 && mod(tgtPos, 2)==0)
    if params.softmaxDim % softmax compression: f(W_h * h_t)
      h2sInfo.input = h_t;
    elseif params.attnFunc % attention
      if params.attnGlobal % soft, global
        srcHidVecs = trainData.absSrcHidVecs;
      else % hard, local
        % positions
        if params.posSignal % unsupervised alignments
          srcPositions = trainData.positions;
        elseif params.predictPos==3 % predict positions by regression
          % h_t -> scales=sigmoid(v_pos*f(W_pos*h_t)) in [0, 1]
          [h2sInfo.scales, h2sInfo.posForwData] = posLayerForward(model.W_pos, model.v_pos, h_t, params);
          
          % h_t -> variances=sigmoid(v_var*f(W_pos*h_t))
          [h2sInfo.variances, h2sInfo.varForwData] = posLayerForward(model.W_var, model.v_var, h_t, params);
          h2sInfo.origVariances = h2sInfo.variances;
          
          % scales -> srcPositions
          h2sInfo.mu = h2sInfo.scales.*trainData.srcLens;
          srcPositions = floor(h2sInfo.mu) + 1;
        else % monotonic alignments
          srcPositions = tgtPos*ones(1, trainData.curBatchSize);
        end
        
        % reverse
        if params.isReverse
          srcPositions = trainData.srcMaxLen - srcPositions;
        end
        
        % build context vectors
        [srcHidVecs, h2sInfo] = buildSrcVecs(trainData.srcHidVecs, srcPositions, trainData.posMask, params, h2sInfo);

        % assert
        if params.assert
          if params.predictPos % use unsupervised alignments
            [srcHidVecs1] = buildSrcVecsOld(params, trainData, srcPositions, trainData.posMask);
            assert(sum(sum(sum(abs(srcHidVecs-srcHidVecs1))))<1e-10);
          else
            [srcHidVecs1, h2sInfo.startAttnId, h2sInfo.endAttnId, h2sInfo.startHidId, h2sInfo.endHidId] = buildSrcHidVecsOld(...
              trainData.srcHidVecs, trainData.srcMaxLen, tgtPos, params);
            assert(sum(sum(sum(abs(srcHidVecs(:, curMask.unmaskedIds)-srcHidVecs1(:, curMask.unmaskedIds)))))<1e-10);
          end
        end   
      end
      
      % f(W_h*[attn_t; tgt_h_t]), attn_t is the context vector in the paper.
      if params.predictPos==3
        if params.isReverse % get back correct source positions
          h2sInfo.indicesAll = trainData.srcMaxLen - h2sInfo.indicesAll;
        end
        
        h2sInfo.alignWeights = zeroMatrix([trainData.curBatchSize, params.numAttnPositions], params.isGPU, params.dataType);
        
        % for computing the guassian probs faster
        h2sInfo.sigAbs = sqrt(h2sInfo.variances);
        h2sInfo.scaledPositions = (h2sInfo.indicesAll-h2sInfo.mu)./h2sInfo.sigAbs;
        if params.isGPU
          h2sInfo.alignWeights(h2sInfo.linearIdSub) = arrayfun(@gaussProb, h2sInfo.scaledPositions, h2sInfo.sigAbs);
        else
          h2sInfo.alignWeights(h2sInfo.linearIdSub) = exp(-0.5*h2sInfo.scaledPositions.^2)./(params.sqrt2pi*h2sInfo.sigAbs);
        end
        
        h2sInfo.alignWeights = permute(h2sInfo.alignWeights, [3, 1, 2]); % 1*curBatchSize*numAttnPositions
        attnVecs = squeeze(sum(bsxfun(@times, srcHidVecs, h2sInfo.alignWeights), 3));
      else
        [attnVecs, h2sInfo.alignWeights] = attnLayerForward(model.W_a, h_t, srcHidVecs, curMask.mask);
      end

      % concat
      h2sInfo.input = [attnVecs; h_t];
      h2sInfo.h_t = h_t;
    end
    
    softmax_h = hiddenLayerForward(model.W_h, h2sInfo.input, params.nonlinear_f);
    h2sInfo.softmax_h = softmax_h;
  else % no intermediate layer
    softmax_h = h_t;
  end
end

% x is already scaled x = (x_orig - mu)/sigAbs
function [prob] = gaussProb(scaledX, sigAbs)
  %prob = exp(-0.5*(x-mu)^2/variance)/sqrt(2*pi*variance);
  prob = exp(-0.5*scaledX^2)/(sqrt(2*pi)*sigAbs);
end
      
%       if params.numAttnPositions>1  
%       else
%         attnVecs = bsxfun(@times, srcHidVecs, curMask.mask);
%         h2sInfo.alignWeights = ones(1, params.curBatchSize).*curMask.mask;
%       end


%         if params.oldSrcVecs % old
%           [srcHidVecs, h2sInfo.linearIndices, h2sInfo.unmaskedIds, h2sInfo.attnLinearIndices] = buildSrcVecsOld(tgtPos, params, trainData, trainData.positions, curMask);
%         else % new
%         end

%           % TODO move this code out
%           if params.attnRelativePos
%             srcPositions = tgtPos - (trainData.positions - params.zeroPosId); % src_pos = tgt_pos - relative_pos
%           else % absolute position
%           end


%     elseif params.posModel==3 && mod(tgtPos, 2)==0 % positional model 3: f(W_h * [srcPosVecs; h_t])
%       if isTest==0
%         positions = trainData.tgtOutput(:, tgtPos-1)'; % Here we look at the previous time steps for positions
%       else
%         positions = trainData.positions;
%       end
%       
%       [srcHidVecs, h2sInfo.linearIndices] = buildSrcPosVecs(tgtPos, params, trainData, positions, curMask);
%       h2sInfo.input = [srcHidVecs; h_t];

%       if params.predictPos % hard attention
%           % h_t -> h_pos
%           [h2sInfo.h_pos] = hiddenLayerForward(model.W_h_pos, h_t, params.nonlinear_f);
%           
%           % predict weight for the src hidden state: sigmoid(v_pos'*f(W_h*h_t))
%           alignScores = model.v_pos*h2sInfo.h_pos;
%           h2sInfo.alignWeights = params.nonlinear_gate_f(alignScores); % 1*batchSize
%           
%           % select alignment vector
%           [srcHidVecs, h2sInfo.linearIndices, h2sInfo.unmaskedIds] = buildSrcPosVecs(tgtPos, params, trainData, trainData.positions, curMask);
%           attnVecs = bsxfun(@times,h2sInfo.alignWeights, srcHidVecs);
%       else % soft attention
%       end
    
