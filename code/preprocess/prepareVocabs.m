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
      params.srcVocab = {'<unk>', '<s>', '</s>', 'x', 'y'};
      % params.srcSos = 2;
    end
    params.tgtVocab = {'<unk>', '<s>', '</s>', 'a', 'b'};
    % params.tgtSos = 2;
    % params.tgtEos = 3;
    
    % char
    if params.charOpt
      assert(params.srcCharShortList == 5);
      assert(params.tgtCharShortList == 5);
      if params.isBi
        params.srcVocab = {'<unk>', '<s>', '</s>', 'x', 'y', 'xy', 'xz', 'z', 'xyz'};
        params.srcCharVocab = {'<c_s>', '</c_s>', 'x', 'y', 'z'};
        params.srcCharMap = {[], [], [], [], [], [3 4], [3 5], 5, [3 4 5]};
      end
      
      params.tgtVocab = {'<unk>', '<s>', '</s>', 'a', 'b', 'ab', 'ac', 'c', 'abc'};
      params.tgtCharVocab = {'<c_s>', '</c_s>', 'a', 'b', 'c'};
      params.tgtCharMap = {[], [], [], [], [], [3 4], [3 5], 5, [3 4 5]};
    end
  else
    % word
    if params.isBi
      [params.srcVocab] = loadVocab(params.srcVocabFile);
    end
    [params.tgtVocab] = loadVocab(params.tgtVocabFile);    
    
    % char
    if params.charOpt
      params.srcCharVocab = loadVocab(params.srcCharVocabFile);
      params.srcCharMap = loadWord2CharMap(params.srcCharMapFile, params.charMaxLen);

      params.tgtCharVocab = loadVocab(params.tgtCharVocabFile);
      params.tgtCharMap = loadWord2CharMap(params.tgtCharMapFile, params.charMaxLen);
    end
  end
  
  isOldVocab = 0;
  if isOldVocab % NOTE: only turn this on to decode old word-based models
    %% src vocab
    if params.isBi
      fprintf(2, '## Bilingual setting\n');
      params.srcVocab{end+1} = '<s_sos>'; % not learn
      params.srcSos = length(params.srcVocab);
      params.srcVocabSize = length(params.srcVocab);
    else
      fprintf(2, '## Monolingual setting\n');
    end

    %% tgt vocab 
    params.tgtVocab{end+1} = '<t_sos>';
    params.tgtSos = length(params.tgtVocab);
    params.tgtVocab{end+1} = '<t_eos>';
    params.tgtEos = length(params.tgtVocab); 
  else
    %% src vocab
    if params.isBi
      fprintf(2, '## Bilingual setting\n');
      params.srcSos = lookup(params.srcVocab, '<s>');
      params.srcUnk = lookup(params.srcVocab, '<unk>');
      assert(~isempty(params.srcSos));

      params.srcVocabSize = length(params.srcVocab);
    else
      fprintf(2, '## Monolingual setting\n');
    end

    %% tgt vocab 
    params.tgtUnk = lookup(params.tgtVocab, '<unk>');
    params.tgtSos = lookup(params.tgtVocab, '<s>');
    params.tgtEos = lookup(params.tgtVocab, '</s>');
    assert(~isempty(params.tgtSos) && ~isempty(params.tgtEos)); 
  end

  %% char
  if params.charOpt
    % sos
    params.srcCharSos = lookup(params.srcCharVocab, '<c_s>');
    if isempty(params.srcCharSos)
      params.srcCharVocab{end+1} = '<c_s>'; % not learn
      params.srcCharSos = length(params.srcCharVocab);
      fprintf(2, '  adding <c_s> %d to src char vocab\n', params.srcCharSos);
    end
    params.tgtCharSos = lookup(params.tgtCharVocab, '<c_s>');
    if isempty(params.tgtCharSos)
      params.tgtCharVocab{end+1} = '<c_s>'; % not learn
      params.tgtCharSos = length(params.tgtCharVocab);
      fprintf(2, '  adding <c_s> %d to tgt char vocab\n', params.tgtCharSos);
    end
    
    % eos
    params.srcCharEos = lookup(params.srcCharVocab, '</c_s>');
    if isempty(params.srcCharEos)
      params.srcCharVocab{end+1} = '</c_s>';
      params.srcCharEos = length(params.srcCharVocab);
      fprintf(2, '  adding </c_s> %d to src char vocab\n', params.srcCharEos);
    end
    params.tgtCharEos = lookup(params.tgtCharVocab, '</c_s>');
    if isempty(params.tgtCharEos)
      params.tgtCharVocab{end+1} = '</c_s>';
      params.tgtCharEos = length(params.tgtCharVocab);
      fprintf(2, '  adding </c_s> %d to tgt char vocab\n', params.tgtCharEos);
    end
    
    params.srcCharVocabSize = length(params.srcCharVocab);
    params.tgtCharVocabSize = length(params.tgtCharVocab);
  end
    
  params.tgtVocabSize = length(params.tgtVocab);
end

function [id] = lookup(vocab, str)
  id = find(strcmp(str, vocab), 1);
end
