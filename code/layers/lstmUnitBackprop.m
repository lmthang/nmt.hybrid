function [dc, dh, d_input, d_W_rnn] = lstmUnitBackprop(W, lstm, c_t_1, dc, dh, maskedIds, params, isFeedInput)
% LSTM unit back prop
% Input:
%   W: recurrent parameters
%   c_t_1: previous cell state
%
% Output:
%   gradients with respect to c, h, input, and the recurrent connenctions
% Thang Luong @ 2014, 2015, <lmthang@stanford.edu>

  dh(:, maskedIds) = 0;
  dc(:, maskedIds) = 0;
  
  if params.isGPU
    %% dh, dc
    if params.lstmOpt==0 % h_t = o_g * f(c_t)
      % dc = dc + f'(c_t)*o_g*dh
      dc = arrayfun(@plusTanhPrimeTriple, dc, lstm.f_c_t, lstm.o_gate, dh);
      % do = f'(o_g)*f(c_t)*dh
      do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, lstm.f_c_t, dh);
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % dc = dc + o_g* dh
      dc = arrayfun(@plusMult, dc, lstm.o_gate, dh);
      % do = f'(o_g)*c_t*dh
      do = arrayfun(@sigmoidPrimeTriple, lstm.o_gate, lstm.c_t, dh); % lstm.c_t
    end

    %% Note: di, df, do, da: w.r.t to i, f, o, a before apply non-linear functions. 
    % di = f'(i_g) * a_signal * dc
    di = arrayfun(@sigmoidPrimeTriple, lstm.i_gate, lstm.a_signal, dc);
    
    % df = f'(f_g)*c_{t-1}*dc
    df = arrayfun(@sigmoidPrimeTriple, lstm.f_gate, c_t_1, dc); % lstm{ll, t-1}.c_t
    
    % da = f'(a_signal)*i_g*dc
    da = arrayfun(@tanhPrimeTriple, lstm.a_signal, lstm.i_gate, dc);
  else
    %% dh, dc
    if params.lstmOpt==0 % h_t = o_g * f(c_t)
      % dc = dc + f'(c_t)*o_g*dh
      dc = dc + params.nonlinear_f_prime(lstm.f_c_t).*lstm.o_gate.*dh;
      % do = f'(o_g)*f(c_t)*dh
      do = params.nonlinear_gate_f_prime(lstm.o_gate).*lstm.f_c_t .* dh;
    elseif params.lstmOpt==1 % h_t = o_g * c_t
      % dc = dc + o_g* dh
      dc = dc + lstm.o_gate.*dh;
      % do = f'(o_g)*c_t*dh
      do = params.nonlinear_gate_f_prime(lstm.o_gate).*lstm.c_t .* dh; % lstm.c_t
    end

    %% Note: di, df, do, da: w.r.t to i, f, o, a before apply non-linear functions. 
    % di = f'(i_g) * a_signal * dc
    di = params.nonlinear_gate_f_prime(lstm.i_gate).*lstm.a_signal.*dc;
    % df = f'(f_g)*c_{t-1}*dc
    df = params.nonlinear_gate_f_prime(lstm.f_gate).*c_t_1.*dc; % lstm{ll, t-1}.c_t

    % da = f'(a_signal)*i_g*dc
    da = params.nonlinear_f_prime(lstm.a_signal).*lstm.i_gate.*dc;   
  end
  
  % update dc
  dc = lstm.f_gate.*dc; % contribute to grad of c_{t-1} = f_t * d(c_t) %(lstm.f_gate + lstm.f_bias)
  
  % grad W
  d_ifoa = [di; df; do; da];
  d_W_rnn = d_ifoa*lstm.input';

  % dx, dh
  d_input = W'*d_ifoa;
 
  % dropout
  if params.dropout<1
    if isFeedInput % t>=srcMaxLen && ll==1 && params.feedInput % predict words
      d_input(1:2*params.lstmSize, :) = d_input(1:2*params.lstmSize, :).*lstm.dropoutMaskInput; % dropout x_t, s_t
    else
      d_input(1:params.lstmSize, :) = d_input(1:params.lstmSize, :).*lstm.dropoutMask; % dropout x_t
    end
%     d_input(1:params.lstmSize, :) = d_input(1:params.lstmSize, :).*lstm.dropoutMask; % dropout x_t
  end
  
  % clip hidden/cell derivatives
  if params.isClip
    if params.isGPU
     d_input = arrayfun(@clipBackward, d_input);
     dc = arrayfun(@clipBackward, dc);
    else
     d_input(d_input>params.clipBackward) = params.clipBackward; d_input(d_input<-params.clipBackward) = -params.clipBackward;
     dc(dc>params.clipBackward) = params.clipBackward; dc(dc<-params.clipBackward) = -params.clipBackward;
    end
  end
  
  dh = d_input(end-params.lstmSize+1:end, :);
  d_input = d_input(1:end-params.lstmSize, :);
  
  % assert
  if params.assert
    assert(computeSum(d_input(:, maskedIds), params.isGPU)==0);
  end
end

function [clippedValue] = clipBackward(x)
  if x>1000
    clippedValue = single(1000);
  elseif x<-1000
    clippedValue = single(-1000);
  else
    clippedValue = x;
  end
end

% compute sigmoid'(x) * y * z = x * (1-x) * y * z
function [value] = sigmoidPrimeTriple(x, y, z)
  value = x*(1-x)*y*z;
end

% compute tanh'(x) * y * z = (1-x*x) * y * z
function [value] = tanhPrimeTriple(x, y, z)
  value = (1-x*x)*y*z;
end

% compute t + tanh'(x) * y * z = t + (1-x*x) * y * z
function [value] = plusTanhPrimeTriple(t, x, y, z)
  value = t + (1-x*x)*y*z;
end

% compute x + y * z
function [value] = plusMult(x, y, z)
  value = x + y*z;
end
