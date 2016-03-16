function [lstmState] = lstmUnitForward(W, x_t, h_t_1, c_t_1, params, rnnFlags)
% LSTM unit
% Input:
%   W: parameter
%   x_t: current input
%   h_t_1, c_t_1: previous hidden state, cell state
%   isTest: 1 -- don't store intermediate results
%
% Output:
%   lstmState struct
% Thang Luong @ 2014, 2015, <lmthang@stanford.edu>

  %% dropout
  if params.dropout<1 && rnnFlags.test==0
    if ~params.isGradCheck
      if params.isGPU
        keepProb = params.dropout;
        invKeepProb = 1/params.dropout;
        dropoutMask = arrayfun(@computeDropoutMask, randMatrix(size(x_t), params.isGPU, params.dataType), keepProb, invKeepProb);
      else
        dropoutMask = (randMatrix(size(x_t), params.isGPU, params.dataType)<params.dropout)/params.dropout;
      end
    else % for gradient check use the same mask
      if rnnFlags.feedInput
        dropoutMask = params.dropoutMaskInput;
      else
        dropoutMask = params.dropoutMask;
      end
    end
    x_t = x_t.*dropoutMask;
  end
  
  %% input, forget, output gates and input signals before applying non-linear functions
  input = [x_t; h_t_1];
  ifoa_linear = W*input; 

  %% cell
  % GPU note: the below non-linear functions are fast, so no need to use arrayfun
  % Updated note: turns out with arrayfun it's even faster, esp. for large models
  if params.isGPU
    ifo_gate = arrayfun(params.nonlinear_gate_f, ifoa_linear(1:3*params.lstmSize, :));
  else
    ifo_gate = params.nonlinear_gate_f(ifoa_linear(1:3*params.lstmSize, :));
  end
  i_gate = ifo_gate(1:params.lstmSize, :);
  f_gate = ifo_gate(params.lstmSize+1:2*params.lstmSize, :);
  o_gate = ifo_gate(2*params.lstmSize+1:3*params.lstmSize, :);
  if params.isGPU
    a_signal = arrayfun(params.nonlinear_f, ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
    c_t = arrayfun(@computeCell, f_gate, c_t_1, i_gate, a_signal);
  else
    a_signal = params.nonlinear_f(ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
    c_t = f_gate.*c_t_1 + i_gate.*a_signal; % c_t = f_t * c_{t-1} + i_t * a_t % (f_gate + f_bias)
  end

  %% hidden
  if params.lstmOpt==0 % h_t = o_t * f(c_t)
    if params.isGPU
      f_c_t = arrayfun(params.nonlinear_f, c_t);
    else
      f_c_t = params.nonlinear_f(c_t);
    end
    
    h_t = o_gate.*f_c_t; 
  elseif params.lstmOpt==1 % h_t = o_t * c_t
    h_t = o_gate.*c_t; 
  end

  %% clip
  if params.isClip
    if params.isGPU
     c_t = arrayfun(@clipForward, c_t);
     h_t = arrayfun(@clipForward, h_t);
    else
     c_t(c_t>params.clipForward) = params.clipForward; c_t(c_t<-params.clipForward) = -params.clipForward; % clip: keep memory small
     h_t(h_t>params.clipForward) = params.clipForward; h_t(h_t<-params.clipForward) = -params.clipForward; % clip: keep hidden state small
    end
  end
  
  
  % assert
  if params.assert
    assert(computeSum(h_t(:, maskedIds), params.isGPU)==0);
    assert(computeSum(c_t(:, maskedIds), params.isGPU)==0);
  end
  
  lstmState.h_t = h_t;
  lstmState.c_t = c_t;
  if (rnnFlags.test==0) % store intermediate results
    lstmState.input = input;
    lstmState.i_gate = i_gate;
    lstmState.f_gate = f_gate;
    lstmState.o_gate = o_gate;
    lstmState.a_signal = a_signal;
    lstmState.f_c_t = f_c_t;
    
    if params.dropout<1 % store dropout mask
      if rnnFlags.feedInput
        lstmState.dropoutMaskInput = dropoutMask;
      else
        lstmState.dropoutMask = dropoutMask;
      end
    end
  end
end

function [mask] = computeDropoutMask(randVal, keepProb, invKeepProb)
  if randVal < keepProb
    mask = invKeepProb;
  else
    mask = 0;
  end
end

function [c_t] = computeCell(f_gate, c_t_1, i_gate, a_signal)
  c_t = f_gate*c_t_1 + i_gate*a_signal;
end

function [clippedValue] = clipForward(x)
  if x>50
    clippedValue = single(50);
  elseif x<-50
    clippedValue = single(-50);
  else
    clippedValue = x;
  end
end