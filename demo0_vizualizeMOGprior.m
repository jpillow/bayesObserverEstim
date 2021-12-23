%  Visualize a Mixture of Gaussians Prior approximating a prior of interest


% --- set parameters of basis Gaussians -------
nbasis = 6; % number of Gaussian basis functions for prior 
bctrs = zeros(1,nbasis);  % prior means
sigshft = 2; % shift the location of first sigma
bsigs = (2.^(0-sigshft:nbasis-1-sigshft)); % prior stdevs

% --- make grid over x axis ------
dx = .1; % bin size
xrnge = 10*[-1 1]; % range of prior to consider
xgrid = (xrnge(1)+dx/2:dx:xrnge(2))';  % grid of stimulus values

% --- Make basis Gaussians -------
basisFun = @(x,mus,sigs)(exp(-0.5*(x-mus).^2./sigs.^2)./(sqrt(2*pi)*sigs));
gbasis = basisFun(xgrid,bctrs,bsigs); % basis centers
gbasis = gbasis./(dx*sum(gbasis)); % normalize so each sums to 1

% --- Set a target prior to approximate ----   
priortype = 3;  % valid options: 1 = gaussian, 2 = exp, 3 = cauchy
switch priortype
    case 1, priortarg = normpdf(xgrid,0,1);   % gaussian
    case 2, priortarg = exp(-abs(xgrid))/2;   % exponential
    case 3, priortarg = (1./(1+xgrid.^2))/pi; % cauchy
end
priortarg = priortarg/sum(priortarg*dx); % normalize the prior to sum to 1

% --- Compute weights on basis functions using quadratic programming -----
v1 = ones(1,nbasis); % vector of ones
v0 = zeros(1,nbasis); % vector of zeros
H = gbasis'*gbasis; % quadratic term
f = -gbasis'*priortarg; % linear term
bwts = quadprog(H,f,[],[],v1,1,v0,v1); % solve for positive weights 

% Compute basis approximation to prior
priorfit = gbasis*bwts;

% ---- Plot basis and prior ------
subplot(221)
plot(xgrid,gbasis); box off;
xlabel('x'); title('basis functions');

subplot(222)
plot(xgrid,priortarg,xgrid,priorfit, '--');
title('prior'); box off;
legend('exact', 'basis approx');
xlabel('x'); ylabel('p(x)');

subplot(223)
semilogy(xgrid,gbasis); box off;
xlabel('x'); title('log basis functions');

subplot(224)
semilogy(xgrid,priortarg,xgrid,priorfit, '--');
title('log prior'); box off;
legend('exact', 'basis approx');
xlabel('x'); ylabel('p(x)');

