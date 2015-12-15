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
    % word
    if params.isBi
      params.srcVocab = {'<s>', '</s>', 'x', 'y'};
    end
    params.tgtVocab = {'<s>', '</s>', 'a', 'b'};
      
    % char
    if params.charShortList
      assert(params.charShortList == 4);
      if params.isBi
        params.srcVocab = {'<s>', '</s>', 'x', 'y', 'xy', 'xz', 'z', 'xyz'};
        params.srcCharVocab = {'x', 'y', 'z'};
        params.srcCharMap = {[], [], [], [], [1 2], [1 3], 3, [1 2 3]};
      end
      
      params.tgtVocab = {'<s>', '</s>', 'a', 'b', 'ab', 'ac', 'c', 'abc'};
      params.tgtCharVocab = {'a', 'b', 'c'};
      params.tgtCharMap = {[], [], [], [], [1 2], [1 3], 3, [1 2 3]};
    end
  else
    % word
    if params.isBi
      [params.srcVocab] = loadVocab(params.srcVocabFile);
    end
    [params.tgtVocab] = loadVocab(params.tgtVocabFile);    
    
    % char
    if params.charShortList
      params.srcCharVocab = loadVocab(params.srcCharVocabFile);
      params.srcCharMap = loadWord2CharMap(params.srcCharMapFile);
      
      params.tgtCharVocab = loadVocab(params.tgtCharVocabFile);
      params.tgtCharMap = loadWord2CharMap(params.tgtCharMapFile);
    else
    end
  end
      
  %% src vocab
  if params.isBi
    fprintf(2, '## Bilingual setting\n');
    params.srcSos = lookup(params.srcVocab, '<s>');
    assert(~isempty(params.srcSos));
    params.srcVocabSize = length(params.srcVocab);
  else
    fprintf(2, '## Monolingual setting\n');
  end
    
  %% tgt vocab 
  params.tgtSos = lookup(params.tgtVocab, '<s>');
  params.tgtEos = lookup(params.tgtVocab, '</s>');
  assert(~isempty(params.tgtSos) && ~isempty(params.tgtEos));
  params.tgtVocabSize = length(params.tgtVocab);
  
  %% char
  if params.charShortList
    assert(params.charShortList < params.srcVocabSize);
    assert(params.charShortList < params.tgtVocabSize);

    params.srcCharVocabSize = length(params.srcCharVocab);
    params.srcRareWordMap = zeros(1, params.srcVocabSize);

    params.tgtCharVocabSize = length(params.tgtCharVocab);
    params.tgtRareWordMap = zeros(1, params.tgtVocabSize);
  end
end

function [id] = lookup(vocab, str)
  id = find(strcmp(str, vocab), 1);
end
