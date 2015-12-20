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
      % params.srcSos = 1;
    end
    params.tgtVocab = {'<s>', '</s>', 'a', 'b'};
    % params.tgtSos = 1;
    % params.tgtEos = 2;
    
    % char
    if params.charShortList
      assert(params.charShortList == 4);
      if params.isBi
        params.srcVocab = {'<s>', '</s>', 'x', 'y', 'xy', 'xz', 'z', 'xyz'};
        params.srcCharVocab = {'<c_s>', '</c_s>', 'x', 'y', 'z'};
        params.srcCharMap = {[], [], [], [], [3 4], [3 5], 5, [3 4 5]};
      end
      
      params.tgtVocab = {'<s>', '</s>', 'a', 'b', 'ab', 'ac', 'c', 'abc'};
      params.tgtCharVocab = {'<c_s>', '</c_s>', 'a', 'b', 'c'};
      params.tgtCharMap = {[], [], [], [], [3 4], [3 5], 5, [3 4 5]};
      
      params.charSos = 1;
      params.charEos = 2;
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
      
      % sos
      srcCharSos = lookup(params.srcCharVocab, '<c_s>');
      tgtCharSos = lookup(params.tgtCharVocab, '<c_s>');
      assert(srcCharSos == tgtCharSos);
      params.charSos = srcCharSos;
      
      % eos
      srcCharEos = lookup(params.srcCharVocab, '</c_s>');
      tgtCharEos = lookup(params.tgtCharVocab, '</c_s>');
      assert(srcCharEos == tgtCharEos);
      params.charEos = srcCharEos;
    else
    end
  end
      
  %% src vocab
  if params.isBi
    fprintf(2, '## Bilingual setting\n');
    params.srcSos = lookup(params.srcVocab, '<s>');
    assert(~isempty(params.srcSos));
%     params.srcVocab{end+1} = '<s_sos>'; % not learn
%     params.srcSos = length(params.srcVocab);
    
    params.srcVocabSize = length(params.srcVocab);
  else
    fprintf(2, '## Monolingual setting\n');
  end
    
  %% tgt vocab 
  params.tgtSos = lookup(params.tgtVocab, '<s>');
  params.tgtEos = lookup(params.tgtVocab, '</s>');
  assert(~isempty(params.tgtSos) && ~isempty(params.tgtEos));
  
%   params.tgtVocab{end+1} = '<t_sos>';
%   params.tgtSos = length(params.tgtVocab);
%   params.tgtVocab{end+1} = '<t_eos>';
%   params.tgtEos = length(params.tgtVocab); 

  params.tgtVocabSize = length(params.tgtVocab);
  
  %% char
  if params.charShortList
    assert(params.charShortList < params.srcVocabSize);
    assert(params.charShortList < params.tgtVocabSize);
    params.srcCharVocabSize = length(params.srcCharVocab);
    params.tgtCharVocabSize = length(params.tgtCharVocab);
    %params.srcRareWordMap = zeros(1, params.srcVocabSize);
    %params.tgtRareWordMap = zeros(1, params.tgtVocabSize);
  end
end

function [id] = lookup(vocab, str)
  id = find(strcmp(str, vocab), 1);
end
