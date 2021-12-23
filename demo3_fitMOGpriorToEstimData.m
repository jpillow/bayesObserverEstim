% Infer the prior underlying a Bayesian observer from estimation data, 
% using a Mixture-of-Gaussians (MOG) parametrization

% ---  Set parameters of the true Bayesian observer ------------

signse = 1.2;  % stdev of observation noise 
xtestrnge = 4*[-1 1]; % range of test stimuli to consider

% --- make grid for stimulus and measurements ----
dx = .1; % bin size
mrnge = xtestrnge + 3*signse*[-1 1]; % range of measurements to consider
xrnge = mrnge + 2*signse*[-1 1]; % range of posterior over x
xgrid = (xrnge(1)+dx/2:dx:xrnge(2))'; % stimulus grid
mgrid = (mrnge(1)+dx/2:dx:mrnge(2))'; % internal measurement grid

% --- Set a prior ----   
priortype = 3;  % valid options: 1 = gaussian, 2 = exp, 3 = cauchy
switch priortype
    case 1, prior = normpdf(xgrid,0,2);   % gaussian
    case 2, prior = exp(-abs(xgrid))/2;   % exponential
    case 3, prior = (1./(1+xgrid.^2))/pi; % cauchy
end
prior = prior/sum(prior*dx); % normalize the prior to sum to 1

%% Compute posterior and BLS estimate for all possible measurements m

% Now compute 2D prior, likelihood & posterior
[xx,mm] = meshgrid(xgrid,mgrid);  % grid of stim and measurement values
ppli = normpdf(mm,xx,signse); % likelihoods
ppost = ppli.*prior'; % unnormalized posterior
ppost = ppost./(sum(ppost,2)*dx); % normalized posterior
BLSestim = ppost*xgrid*dx; % BLS estimate for each m value

%% Simulate data from the model 

nsmps = 1000; % # of samples
xdat = rand(nsmps,1)*diff(xtestrnge)+xtestrnge(1);  % stimuli
mdat = xdat+randn(nsmps,1)*signse; % observer's noisy measurements

% observer's estimate (assuming BLS)
xhat = interp1(mgrid,BLSestim,mdat,'linear','extrap');

%% Set up basis of Gaussians for MOG prior

% --- set parameters of basis Gaussians -------
nbasis = 6; % number of Gaussian basis functions for prior 
bctrs = zeros(1,nbasis);  % prior means
sigshft = 2; % shift the location of first sigma
bsigs = (2.^(0-sigshft:nbasis-1-sigshft)); % prior stdevs

% --- Make basis Gaussians -------
basisFun = @(x,mus,sigs)(exp(-0.5*(x-mus).^2./sigs.^2)./(sqrt(2*pi)*sigs));
gbasis = basisFun(xgrid,bctrs,bsigs); % basis centers
gbasis = gbasis./(dx*sum(gbasis)); % normalize so each sums to 1


%% Now fit the model
[signsehat,priorhat,bwtshat,logliFinal,Mposthat] = fitBLSobserverModel_estimdata(xdat,xhat,gbasis,xgrid,mgrid);

% inferred BLS estimate for each m value
BLSestimhat = Mposthat*xgrid*dx; 


%% Make plots

% ----- Plot true and inferred prior --------

subplot(221) % linear scale
plot(xgrid,prior,xgrid,priorhat, '--');
title('prior'); box off;
legend('true prior', 'inferred prior');
xlabel('x'); ylabel('p(x)');

subplot(223) % log scale
semilogy(xgrid,prior,xgrid,priorhat, '--');
title('log-scale prior'); box off;
legend('true prior', 'inferred prior');
xlabel('x'); ylabel('p(x)');

% ----- Plot true and inferred posterior & BLS estimates --------

subplot(222);  % true posterior
imagesc(xgrid,mgrid,ppost); axis image; axis xy;
title('true posterior');
hold on;
plot(BLSestim,mgrid,'r'); axis image; 
plot(xgrid,xgrid,'k--','linewidth',2);
hold off;
axis([xrnge mrnge]);
legend('true BLS estimate', 'location', 'northwest');
ylabel('measurement m');

subplot(224); % inferred posterior
imagesc(xgrid,mgrid,Mposthat); axis image; axis xy;
title('inferred posterior');
hold on;
plot(BLSestim,mgrid,'r',BLSestimhat,mgrid,'c--'); axis image; 
plot(xgrid,xgrid,'k--','linewidth',2);
hold off;
axis([xrnge mrnge]);
legend('true BLS', 'inferred BLS', 'location', 'northwest');
xlabel('stimulus x');  ylabel('measurement m');
