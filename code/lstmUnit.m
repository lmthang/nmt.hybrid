%% LSTM unit
% Thang Luong @ 2014, <lmthang@stanford.edu>
%
% Input:
%   W: parameter
%   x_t: current input
%   h_t, c_t: previous hidden state, cell state
%   isTest: 1 -- don't store intermediate results
%
% Output:
%   lstm struct
%%
function [lstmCell] = lstmUnit(W, x_t, h_t, c_t, ll, t, srcMaxLen, params, isTest)
  %% dropout
  if params.dropout<1 && isTest==0
    if ~params.isGradCheck
      if params.isGPU
        dropoutMask = (rand(size(x_t), 'gpuArray')<params.dropout)/params.dropout;
      else
        dropoutMask = (rand(size(x_t))<params.dropout)/params.dropout;
      end
    else % for gradient check use the same mask
      if t>=srcMaxLen && ll==1 && params.posModel>0
        dropoutMask = params.dropoutMaskPos;
      else
        dropoutMask = params.dropoutMask;
      end
    end
    
    x_t = x_t.*dropoutMask;
  end
  
  %% input, forget, output gates and input signals before applying non-linear functions
  input = [x_t; h_t];
  ifoa_linear = W*input; 

  %% cell
  % GPU note: the below non-linear functions are fast, so no need to use arrayfun
  ifo_gate = params.nonlinear_gate_f(ifoa_linear(1:3*params.lstmSize, :));
  i_gate = ifo_gate(1:params.lstmSize, :);
  f_gate = ifo_gate(params.lstmSize+1:2*params.lstmSize, :);
  o_gate = ifo_gate(2*params.lstmSize+1:3*params.lstmSize, :);
  a_signal = params.nonlinear_f(ifoa_linear(3*params.lstmSize+1:4*params.lstmSize, :)); % note input uses a different activation function
  lstmCell.c_t = f_gate.*c_t + i_gate.*a_signal; % c_t = f_t * c_{t-1} + i_t * a_t % (f_gate + f_bias)

  %% hidden
  if params.lstmOpt==0 % h_t = o_t * f(c_t)
    f_c_t = params.nonlinear_f(lstmCell.c_t);
    lstmCell.h_t = o_gate.*f_c_t; 
  elseif params.lstmOpt==1 % h_t = o_t * c_t
    lstmCell.h_t = o_gate.*lstmCell.c_t; 
  end

  %% clip
  if params.isClip
    if params.isGPU
     lstmCell.c_t = arrayfun(@clipForward, lstmCell.c_t);
     lstmCell.h_t = arrayfun(@clipForward, lstmCell.h_t);
    else
     lstmCell.c_t(lstmCell.c_t>params.clipForward) = params.clipForward; lstmCell.c_t(lstmCell.c_t<-params.clipForward) = -params.clipForward; % clip: keep memory small
     lstmCell.h_t(lstmCell.h_t>params.clipForward) = params.clipForward; lstmCell.h_t(lstmCell.h_t<-params.clipForward) = -params.clipForward; % clip: keep hidden state small
    end
  end
  
  if (isTest==0) % store intermediate results
    lstmCell.input = input;
    lstmCell.i_gate = i_gate;
    lstmCell.f_gate = f_gate;
    lstmCell.o_gate = o_gate;
    lstmCell.a_signal = a_signal;
    lstmCell.f_c_t = f_c_t;
    
    if params.dropout<1 % store dropout mask
      if t>=srcMaxLen && ll==1 && params.posModel>0
        lstmCell.dropoutMaskPos = dropoutMask;
      else
        lstmCell.dropoutMask = dropoutMask;
      end
    end
  end
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
