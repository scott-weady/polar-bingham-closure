%------------------------------------------------------------------------
% Polar Bingham closure in two dimensions
% Input: 
%   n - first moment tensor, struct with fields n1,n2
%   Q - second moment tensor, struct with fields q11,q12,q22
%   T - rotation tensor T = E + 2*zeta*c*Q, struct with fields t11,t12,t22
%   coeffs - Chebyshev coefficient tensors, struct with fields cr111,cr112,cs1111,cs1112
% Output: 
%   R - third moment tensor, struct with fields r111,r112,r122,r222
%   ST - contraction S:T, struct with fields st11,st12,st22
%
% Scott Weady, CIMS
% Last updated Feb 2022
%------------------------------------------------------------------------
function [R,ST] = pbingham2d(n,Q,T,coeffs)
    
  % Get components
  n1 = n.n1; n2 = n.n2; %first moment
  q11 = Q.q11; q12 = Q.q12; %second moment
  t11 = T.t11; t12 = T.t12; t22 = T.t22; %rotation tensor
  [N,~] = size(n1);
  
  % Eigendecomposition of Q
  mu1 = 0.5*(1 + 2*sqrt(q11.^2 - q11 + q12.^2 + 0.25));
  mu2 = 1 - mu1;
  ohm = 0.5*atan2(2*q12,2*q11-1);
  O11 = cos(ohm); O12 = -sin(ohm);
  O21 = -O12; O22 = O11;
  
  % Transform to Chebyshev grid
  tn1 = O11.*n1 + O21.*n2;
  tn2 = O12.*n1 + O22.*n2;

  r1 = tn1./max(tn1,sqrt(mu1));
  r2 = tn2./max(tn2,sqrt(mu2));
  rr = r1.^2 - r2.^2;
  
  x = 0.5*sqrt(abs(2 + rr + 2*sqrt(2)*r1)) - 0.5*sqrt(abs(2 + rr - 2*sqrt(2)*r1));
  y = 0.5*sqrt(abs(2 - rr + 2*sqrt(2)*r2)) - 0.5*sqrt(abs(2 - rr - 2*sqrt(2)*r2));
  z = 4*mu1 - 3;

  % Evaluate Chebyshev interpolants
  [tr111,tr112,ts1111,ts1112] = chebinterp(x(:),y(:),z(:),coeffs);     
  tr111 = reshape(tr111,[N N]); 
  tr112 = reshape(tr112,[N N]);
  ts1111 = reshape(ts1111,[N N]);
  ts1112 = reshape(ts1112,[N N]);

  % Compute remaining terms 
  tr122 = tn1 - tr111;
  tr222 = tn2 - tr112;

  ts1122 = mu1 - ts1111;
  ts1222 = -ts1112;
  ts2222 = mu2 - ts1122;

  % Rotate R to original coordinate system
  r111 = (O11.*O11.*O11.*tr111 + 3*O11.*O11.*O12.*tr112 + ...
          3*O11.*O12.*O12.*tr122 + O12.*O12.*O12.*tr222); 
  r222 = (O21.*O21.*O21.*tr111 + 3*O21.*O21.*O22.*tr112 + ...
          3*O21.*O22.*O22.*tr122 + O22.*O22.*O22.*tr222); 
  r112 = n2 - r222; 
  r122 = n1 - r111;

  % Rotate T to diagonal coordinate system
  tt11 = O11.*O11.*t11 + 2*O11.*O21.*t12 + O21.*O21.*t22;
  tt12 = O11.*O12.*t11 + (O11.*O22 + O21.*O12).*t12 + O21.*O22.*t22;
  tt22 = O12.*O12.*t11 + 2*O12.*O22.*t12 + O22.*O22.*t22;   

  % Compute contraction tS:tT
  tst11 = ts1111.*tt11 + 2*ts1112.*tt12 + ts1122.*tt22;
  tst12 = ts1112.*tt11 + 2*ts1122.*tt12 + ts1222.*tt22;
  tst22 = ts1122.*tt11 + 2*ts1222.*tt12 + ts2222.*tt22;

  % Rotate tS:tT to original coordinate system
  st11 = O11.*O11.*tst11 + 2*O11.*O12.*tst12 + O12.*O12.*tst22;
  st12 = O11.*O21.*tst11 + (O11.*O22 + O21.*O12).*tst12 + O12.*O22.*tst22;
  st22 = O21.*O21.*tst11 + 2*O21.*O22.*tst12 + O22.*O22.*tst22;
      
  % Store fields
  R = struct('r111',r111,'r112',r112,'r122',r122,'r222',r222);
  ST = struct('st11',st11,'st12',st12,'st21',st12,'st22',st22);
    
end

%------------------------------------------------------------------------
% Evaluates Chebyshev interpolants for tr111,tr112,ts1111,ts1112
%------------------------------------------------------------------------
function [tr111,tr112,ts1111,ts1112] = chebinterp(x,y,z,coeffs)
  
  N = length(x);
  cr111 = coeffs.cr111;
  cr112 = coeffs.cr112;
  cs1111 = coeffs.cs1111;
  cs1112 = coeffs.cs1112;
  
  x = min(1,max(-1,x));
  y = min(1,max(-1,y));
  z = min(1,max(-1,z));
  
  tr111 = zeros(N,1);
  tr112 = zeros(N,1);
  ts1111 = zeros(N,1);
  ts1112 = zeros(N,1);
  [~,M] = size(cs1111);
  m1 = (M-1)/2; m2 = (M-1)/2;

  parfor(n = 1:N)
    Ti = chebeval(x(n),M)'; %compute Chebyshev polynomials in x
    Tj = chebeval(y(n),M); % " " y
    Tk = chebeval(z(n),M); % " " z
    Ti_o = Ti(2:2:end); Ti_e = Ti(1:2:end-1); %get odd and even components
    Tj_o = Tj(2:2:end); Tj_e = Tj(1:2:end-1); % " "
    tr111(n) = Ti_o*(reshape(cr111*Tk,[m1 m2])*Tj_e); %evaluate
    tr112(n) = Ti_e*(reshape(cr112*Tk,[m1 m2])*Tj_o); % " "
    ts1111(n) = Ti_e*(reshape(cs1111*Tk,[m1 m2])*Tj_e); % " "
    ts1112(n) = Ti_o*(reshape(cs1112*Tk,[m1 m2])*Tj_o); % " "
  end

end

%------------------------------------------------------------------------
% Computes the first M Chebyshev polynomials evaluated at x
%------------------------------------------------------------------------
function T = chebeval(x,M)

  T = zeros(M,length(x));  
  T(1,:) = 1; T(2,:) = x;
  
  for m = 3:M
    T(m,:) = 2*x.*T(m-1,:) - T(m-2,:);
  end
  
end
