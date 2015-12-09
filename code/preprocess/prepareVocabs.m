function [params] = prepareVocabs(params)
% prepareVocabs - prepare vocabs for the model
%
% Input:
%   params: parameter settings
%
% Output:
%   params: updated parameter settings with vocab fields
%
% Authors: 
%   Thang Luong @ 2015, <lmthang@stanford.edu>
%

  %% grad check
  if params.isGradCheck
    tgtVocab = {'a', 'b'};

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
    params.srcOrigVocabSize = length(srcVocab);
    
    % add sos, eos not learn
    srcVocab{end+1} = '<s_sos>';
    params.srcSos = length(srcVocab);
    srcVocab{end+1} = '<s_eos>';
    params.srcEos = length(srcVocab);
    
    % char
    if params.charShortList
      srcVocab{end+1} = '<s_rare>';
      params.srcRare = length(srcVocab);
    end
    
    params.srcVocabSize = length(srcVocab);
  else
    fprintf(2, '## Monolingual setting\n');
  end
    
  %% tgt vocab  
  params.tgtOrigVocabSize = length(tgtVocab);
  
  % add eos, sos
  tgtVocab{end+1} = '<t_sos>';
  params.tgtSos = length(tgtVocab);
  tgtVocab{end+1} = '<t_eos>';
  params.tgtEos = length(tgtVocab);
  
    % char
  if params.charShortList
    tgtVocab{end+1} = '<t_rare>';
    params.tgtRare = length(tgtVocab);
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
end

%     params.srcSos = lookup(srcVocab, '<s>');
%     assert(~isempty(params.srcSos));

%   params.tgtSos = lookup(tgtVocab, '<s>');
%   params.tgtEos = lookup(tgtVocab, '</s>');
%   assert(~isempty(params.tgtSos) && ~isempty(params.tgtEos));

% function [id] = lookup(vocab, str)
%   id = find(strcmp(str, vocab), 1);
% end

%     tgtVocab = {'a', 'b', '<s>', '</s>'};
% 
%     if params.isBi
%       srcVocab = {'x', 'y', '<s>', '</s>'};
%     end
