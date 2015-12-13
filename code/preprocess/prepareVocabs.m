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
    tgtVocab = {'<s>', '</s>', 'a', 'b'};

    if params.isBi
      srcVocab = {'<s>', '</s>', 'x', 'y'};
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
    params.srcSos = lookup(srcVocab, '<s>');
    assert(~isempty(params.srcSos));
    params.srcVocabSize = length(srcVocab);
  else
    fprintf(2, '## Monolingual setting\n');
  end
    
  %% tgt vocab 
  params.tgtSos = lookup(tgtVocab, '<s>');
  params.tgtEos = lookup(tgtVocab, '</s>');
  assert(~isempty(params.tgtSos) && ~isempty(params.tgtEos));
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

function [id] = lookup(vocab, str)
  id = find(strcmp(str, vocab), 1);
end
