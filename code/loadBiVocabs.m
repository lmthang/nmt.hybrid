function [params] = loadBiVocabs(params)
  %% grad check
  if params.isGradCheck
    if params.posModel>=0
      params.posWin = 2;
      tgtVocab = {'a', 'b', '<p_-2>', '<p_-1>', '<p_0>', '<p_1>', '<p_2>'};
    else
      tgtVocab = {'a', 'b'};
    end
    
    if params.isBi
      srcVocab = {'x', 'y'};
    end
  else
    [tgtVocab] = loadVocab(params.tgtVocabFile);    
    if params.isBi
      [srcVocab] = loadVocab(params.srcVocabFile);
    end
  end
  
  %% src vocab
  if params.isBi
    fprintf(2, '## Bilingual setting\n');
    
    % add eos, sos, zero
    srcVocab{end+1} = '<s_sos>'; % not learn
    params.srcZero = length(srcVocab);
    srcVocab{end+1} = '<s_eos>';
    params.srcEos = length(srcVocab);
    
    % here we have src eos, so we don't need tgt sos.
    params.srcVocabSize = length(srcVocab);
  else
    fprintf(2, '## Monolingual setting\n');
  end
    
  %% tgt vocab  
  % positional vocab
  if params.posModel>0
    indices = find(strncmp('<p_', tgtVocab, 3));
    assert(length(indices) == (indices(end)-indices(1)+1)); % make sure indices are contiguous
    params.startPosId = indices(1);
    
    pattern = '<p_(.+)>';
    prevPos = -1;
    for ii=1:length(indices)
      n = regexp(tgtVocab{indices(ii)}, pattern, 'tokens');
      pos = str2double(n{1}{1});
      %fprintf(2, '%s\t%d\n', tgtVocab{indices(ii)}, pos);
      
      % zero
      if (pos==0)
        params.zeroPosId = indices(ii);
      end
      
      % assert
      assert(~isnan(pos));
      assert(ii==1 || pos==(prevPos+1));
      prevPos = pos;
    end
    
    params.posVocabSize = length(indices) + 1; % include <eos>
    fprintf(2, '# Positional model: posVocabSize=%d, startPosId=%d, zeroPosId=%d\n', params.posVocabSize, params.startPosId, params.zeroPosId);
    fprintf(params.logId, '# Positional model: posVocabSize=%d, startPosId=%d, zeroPosId=%d\n', params.posVocabSize, params.startPosId, params.zeroPosId);
    
    % NOTE: purposely add eos first, then sos, so that the positional vocab
    % (including eos) is contiguous
    tgtVocab{end+1} = '<t_eos>';
    params.tgtEos = length(tgtVocab);
    tgtVocab{end+1} = '<t_sos>';
    params.tgtSos = length(tgtVocab);
  else
    % add eos, sos
    tgtVocab{end+1} = '<t_sos>';
    params.tgtSos = length(tgtVocab);
    tgtVocab{end+1} = '<t_eos>';
    params.tgtEos = length(tgtVocab);
  end
  params.tgtVocabSize = length(tgtVocab);
  
  
  %% finalize vocab
  if params.isBi
    params.srcVocab = srcVocab;
  else
    %params.inVocabSize = params.tgtVocabSize;
    params.srcVocab = [];
  end
  params.tgtVocab = tgtVocab;
  %params.outVocabSize = params.tgtVocabSize;
  
  if params.assert
    if params.posModel>=1
      assert(params.tgtEos == (params.startPosId + params.posVocabSize-1));
    end
  end
end


%% class-based softmax %%
% beginning
%       if params.numClasses>0
%         tgtVocab = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};
%       else
%         tgtVocab = {'a', 'b'};
%       end
% before adding sos, eos to tgtVocab
%   % class-based softmax
%   if params.numClasses>0 % make sure vocab size is divisible by numClasses
%     remain = params.numClasses - mod(length(tgtVocab)+2, params.numClasses); % assume we have added <sos>, <eos>
%     for ii=1:remain
%       tgtVocab{end+1} = ['<dummy', num2str(ii), '>'];
%     end
%     fprintf('# Using class-based softmax, numClasses=%d, adding %d dummy words, tgt vocab size now = %d\n', params.numClasses, remain, length(tgtVocab)+1);
%   end

%% Unused
%     if params.separateEmb==0
%       params.vocab = [tgtVocab srcVocab];
%       params.srcEos = params.srcEos + params.tgtVocabSize;
%       params.srcZero = params.srcZero + params.tgtVocabSize;
%       params.inVocabSize = params.tgtVocabSize + params.srcVocabSize;
%     end

