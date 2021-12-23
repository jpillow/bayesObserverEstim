%  Simulate some estimation data from a Bayesian ideal observer model

% ---  Set parameters of the true Bayesian observer ------------

signse = 1.5;  % stdev of observation noise 
xtestrnge = 4*[-1 1]; % range of test stimuli to consider

% --- make grid for stimulus and measurements ----
dx = .1; % bin size
mrnge = xtestrnge + 2*signse*[-1 1]; % range of measurements to consider
xrnge = mrnge + 2*signse*[-1 1]; % range of posterior over x
xgrid = (xrnge(1)+dx/2:dx:xrnge(2))'; % stimulus grid
mgrid = (mrnge(1)+dx/2:dx:mrnge(2))'; % internal measurement grid

% --- Set a prior ----   
priortype = 3;  % valid options: 1 = gaussian, 2 = exp, 3 = cauchy
switch priortype
    case 1, prior = normpdf(xgrid,0,2.5);   % gaussian
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

nsmps = 500; % # of samples
xdat = rand(nsmps,1)*diff(xtestrnge)+xtestrnge(1);  % stimuli
mdat = xdat+randn(nsmps,1)*signse; % observer's noisy measurements

% observer's estimate (assuming BLS)
xhat = interp1(mgrid,BLSestim,mdat,'linear','extrap');

%% Make plots

subplot(131);
plot(xdat,mdat,'.'); 
axis image;
set(gca,'xlim',xrnge,'ylim',mrnge);
xlabel('true x'); ylabel('m samples');
title('measurement vs true stim');

subplot(132);
plot(xhat,mdat,'.',xgrid,xgrid,'k');
%hold on; plot(BLSestim,m,'y--','linewidth',1); hold off;
axis image;
set(gca,'xlim',xrnge,'ylim',mrnge);
xlabel('x estimates'); ylabel('m samples');
title('measurement vs stim estimate');

subplot(133);
plot(xdat,xhat,'.',xgrid,xgrid,'k');
axis image;
set(gca,'xlim',xrnge,'ylim',mrnge);
xlabel('true x'); ylabel('x estimates'); 
title('stim estimate vs true stim');