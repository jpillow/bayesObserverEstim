function [signse,prihat,bwtshat,Mlihat,Mposthat] = fitBLSobserverModel_estimdata(x,xhat,Pbasis,xgrid,mgrid,prs0)
% [signse,prihat,bwtshat,Mlihat,Mposthat] = fitBLSobserverModel_estimdata(x,xhat,Pbasis,xgrid,mgrid,prs0)
%
% Fit Bayesian ideal observer model with fixed Gaussian noise from observer
% estimate data, consisting of pairs of stimuli 'x' and observer estimates
% 'x_hat'.
%
% Inputs:
% -------
%           x [N x 1]  - column vector of presented stimuli
%       x_hat [N x 1]  - column vector of observer estimates
%     Pbasis [Nx x Nb] - matrix of basis vectors for representing the prior
%                         (each column should sum to 1)
%       xgrid [Nx x 1] - grid of x evenly-spaced values on which prior is defined
%       mgrid [Nx x 1] - grid of "measurement" m values with same spacing
%                         (though can have larger range)
%       prs0 [Nb+1 x 1]- initial parameters: [signse0; bwts0] (OPTIONAL)
%
% Output:
% -------
%    signse - estimate of stdev of encoding noise
%    prihat - estimated prior on x grid
%      bwts - estimated weights of prior in basis
%    Mlihat - matrix of inferred likelihood 
%  Mposthat - matrix of inferred posterior


[nX,nB] = size(Pbasis);

if nX ~= length(xgrid);
    error('mismatch in xgrid and size of basis for prior');
end

dx = diff(xgrid(1:2));
Pbasis = bsxfun(@times,Pbasis,1./sum(Pbasis))/dx;

% Make sure vector arguments are column vectors
x = x(:); xhat=xhat(:); xgrid=xgrid(:); mgrid=mgrid(:);

if nargin<6
    wts0 = normpdf(1:nB,(nB+1)/2,nB/4)';
    wts0 = wts0./sum(wts0);
    signse0 = std(x-xhat); % initial estimate of noise variance
    prs0 = [signse0;wts0];
end

Dat = [x,xhat];

% bounds for integration
LB = [1e-2; zeros(nB,1)]; % lower bound
UB = [1e3;ones(nB,1)];  % upper bound
Aeq = [0, ones(1,nB)]; % equality constraint (prior sums to 1)
beq = 1;  % equality constraint)

lossfun = @(prs)(neglogli_BaysObsModel(prs,Dat,Pbasis,xgrid,mgrid));

%opts = optimset('display', 'iter','algorithm','interior-point'); 
opts = optimset('display', 'iter','algorithm','sqp'); 
prshat = fmincon(lossfun,prs0,[],[],Aeq,beq,LB,UB,[],opts);

signse = prshat(1);
bwtshat = prshat(2:end);
prihat = Pbasis*bwtshat;

if nargout > 3
    [Lval,Mlihat,Mposthat,mvals] = lossfun(prshat);
end

% ------- LOSS FUNCTION: negative log-likelihood ---------------
function [L,Mli,Mpost,mvals] = neglogli_BaysObsModel(prs,Dat,Pbasis,xgrid,mgrid)
% Computes negative log-likelihood of data (for optimization)


% finv = @(x)interp1(BLSestim,m,x,'linear','extrap');
% dfinv = @(x)(interp1(BLSestim,finitediff(m)./finitediff(BLSestim),x,'linear','extrap'));

dx = diff(xgrid(1:2));
npts = size(Dat,1);

sig = max(prs(1),1e-2); % make sure noise stdev is positive
wprs = prs(2:end);
wprs = min(max(wprs,0),1);  % enforce that weights \in [0,1]
wprs = wprs./sum(wprs);  % enforce equality constraint

% Compute prior
p = Pbasis*wprs;  

% Compute likelihood
Mli = exp(-bsxfun(@plus,mgrid,-xgrid').^2./(2*sig.^2));  % can ignore the 1./sqrt(2*pi*sig^2) here

% Compute posterior
Mpost = diag(1./(Mli*p*dx))*Mli*diag(p);
Mpost(isnan(Mpost))=0;

% Compute BLS function
xBLS = Mpost*xgrid*dx;

% Prune any zeros or non-increasing regions (can happen when sig is small);
if any(xBLS==0);
    ii = xBLS==0;
    xBLS(ii)=[];
    mgrid(ii)=[];
end
while any(diff(xBLS)<=0)
    ii = (diff(xBLS)<=0);
    xBLS(ii)=[];
    mgrid(ii)=[];
end

% Compute m value for each reported xhat
mvals = interp1(xBLS,mgrid,Dat(:,2),'linear','extrap');


dxhat = interp1(xBLS,finitediff(mgrid)./finitediff(xBLS),Dat(:,2),'linear','extrap');
dxhat = max(dxhat,1e-100);

% Compute negative log-likelihood of data

L = sum((mvals-Dat(:,1)).^2)/(2*sig.^2) + npts*log(sig) - sum(log(dxhat));


