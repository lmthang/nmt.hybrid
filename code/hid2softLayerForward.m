function [softmax_h, h2sInfo] = hid2softLayerForward(h_t, params, model, trainData, tgtPos)
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
          [mu, h2sInfo] = regressPositions(model, h_t, trainData.srcLens, params);
          srcPositions = floor(mu) + 1;
        else % monotonic alignments
          srcPositions = tgtPos*ones(1, trainData.curBatchSize);
        end
        
        % reverse
        if params.isReverse
          srcPositions = trainData.srcMaxLen - srcPositions;
        end
        
        % build context vectors
        [srcHidVecs, h2sInfo] = buildSrcVecs(trainData.srcHidVecs, srcPositions, trainData.posMask, params, h2sInfo);
        
        
        if params.attnGlobal==0 && params.attnOpt==1
          trainData.alignMask = h2sInfo.alignMask;
        end
      end % end else if attnGlobal
      
      % f(W_h*[attn_t; tgt_h_t]), attn_t is the context vector in the paper.
      if params.predictPos==3
        [distWeights, h2sInfo] = gaussLayerForward(mu, h2sInfo, trainData, params);
        
        if params.attnOpt==0 % no src state comparison
          h2sInfo.alignWeights = distWeights; % numAttnPositions*curBatchSize
        elseif params.attnOpt==1 % src state comparison
          h2sInfo.alignWeights = compareSrcStates(srcHidVecs, h_t, trainData.alignMask, params) .* distWeights; % weighted by distances
        end
      else
        if params.attnOpt==0 % no src state comparison
          h2sInfo.alignWeights = softmax(model.W_a*h_t); % numAttnPositions*curBatchSize
        elseif params.attnOpt==1 % src state comparison
          [h2sInfo.alignWeights] = compareSrcStates(srcHidVecs, h_t, trainData.alignMask, params);
        end
      end
      
      % alignWeights, srcHidVecs -> contextVecs
      [contextVecs] = contextLayerForward(h2sInfo.alignWeights, srcHidVecs);
      
      % assert
      if params.assert
        if params.attnGlobal && params.attnOpt==1
          assert(isequal(size(h2sInfo.alignWeights), [params.numSrcHidVecs, params.curBatchSize]));
        else
          assert(isequal(size(h2sInfo.alignWeights), [params.numAttnPositions, params.curBatchSize]));
        end
        assert(isequal(size(h_t), size(contextVecs))); % lstmSize * curBatchSize
      end

      % concat
      h2sInfo.input = [contextVecs; h_t];
      h2sInfo.h_t = h_t;
    end
    
    softmax_h = hiddenLayerForward(model.W_h, h2sInfo.input, params.nonlinear_f);
    h2sInfo.softmax_h = softmax_h;
  else % no intermediate layer
    softmax_h = h_t;
  end
end

function [mu, h2sInfo] = regressPositions(model, h_t, srcLens, params)
  % h_t -> scales=sigmoid(v_pos*f(W_pos*h_t)) in [0, 1]
  [h2sInfo.scales, h2sInfo.posForwData] = scaleLayerForward(model.W_pos, model.v_pos, h_t, params);

  % h_t -> variances=sigmoid(v_var*f(W_pos*h_t))
  [h2sInfo.origVariances, h2sInfo.varForwData] = scaleLayerForward(model.W_var, model.v_var, h_t, params);
  %h2sInfo.origVariances = h2sInfo.variances;

  % scales -> srcPositions
  mu = h2sInfo.scales.*srcLens;
end


%   h2sInfo.alignWeights = permute(h2sInfo.alignWeights, [3, 1, 2]); % 1*curBatchSize*numAttnPositions
%   contextVecs = squeeze(sum(bsxfun(@times, srcHidVecs, h2sInfo.alignWeights), 3));

%         % assert
%         if params.assert
%           if params.predictPos % use unsupervised alignments
%             [srcHidVecs1] = buildSrcVecsOld(params, trainData, srcPositions, trainData.posMask);
%             assert(sum(sum(sum(abs(srcHidVecs-srcHidVecs1))))<1e-10);
%           else
%             [srcHidVecs1, h2sInfo.startAttnId, h2sInfo.endAttnId, h2sInfo.startHidId, h2sInfo.endHidId] = buildSrcHidVecsOld(...
%               trainData.srcHidVecs, trainData.srcMaxLen, tgtPos, params);
%             assert(sum(sum(sum(abs(srcHidVecs(:, curMask.unmaskedIds)-srcHidVecs1(:, curMask.unmaskedIds)))))<1e-10);
%           end
%         end   


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
    
