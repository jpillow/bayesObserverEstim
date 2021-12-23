%  Visualize some data simulated from a Mixture of Gaussians Prior

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
    case 1, prior = normpdf(xgrid,0,1);   % gaussian
    case 2, prior = exp(-abs(xgrid))/2;   % exponential
    case 3, prior = (1./(1+xgrid.^2))/pi; % cauchy
end
prior = prior/sum(prior*dx); % normalize the prior to zum to 1

% Pick some measurement values at which to plot the posterior
mcond = [1 2 4];  

%% Compute matrices for display and show posterior

% Now compute 2D prior, likelihood & posterior
[xx,mm] = meshgrid(xgrid,mgrid);  % grid of stim and measurement values
ppli = normpdf(mm,xx,signse); % likelihoods
ppost = ppli.*prior'; % unnormalized posterior
ppost = ppost./(sum(ppost,2)*dx); % normalized posterior

% compute MAP estimator and BLS estimators for each value of m
[~,imap] = max(ppost,[],2);
MAPestim = xgrid(imap);    % MAP estimate for each m value (if desired)
BLSestim = ppost*xgrid*dx; % BLS estimate for each m value

% Make plots
xlm = xrnge;
subplot(231);
imagesc(xgrid,mgrid,prior'); axis image; axis xy;
xlabel('x'); ylabel('m');title('prior');
axis([xlm mrnge]);
subplot(232);
imagesc(xgrid,mgrid,ppli); axis image; axis xy; 
xlabel('x'); title('likelihood');
axis([xlm mrnge]);
subplot(233);
imagesc(xgrid,mgrid,ppost); axis image; axis xy;
xlabel('x'); title('posterior');
hold on;
plot(BLSestim,mgrid,'r'); axis image; 
plot(xgrid,xgrid,'k--','linewidth',2);
hold off;
axis([xlm mrnge]);
legend('BLS estimate', 'location', 'northwest');

% Plot a few slices (posteriors given particular values of m)
xlm = xrnge; ylm = [0 max(ppost(:))*1.05];
subplot(234);
[~,i1] = min(abs(mgrid-mcond(1)));
plot(xgrid,prior,xgrid,ppli(i1,:),xgrid,ppost(i1,:));
box off; set(gca,'xlim',xlm,'ylim',ylm);
title(sprintf('m = %.1f',mgrid(i1)));
xlabel('x'); ylabel('p(x)');
subplot(235);% ---------
[~,i1] = min(abs(mgrid-mcond(2)));
plot(xgrid,prior,xgrid,ppli(i1,:),xgrid,ppost(i1,:));
box off; set(gca,'xlim',xlm,'ylim',ylm);
title(sprintf('m = %.1f',mgrid(i1)));
xlabel('x');
subplot(236); % ---------
[~,i1] = min(abs(mgrid-mcond(3)));
plot(xgrid,prior,xgrid,ppli(i1,:),xgrid,ppost(i1,:));
box off; set(gca,'xlim',xlm,'ylim',ylm);
title(sprintf('m = %.1f',mgrid(i1)));
legend('prior','likelihood','posterior');
xlabel('x');
% subplot(234);
% xlabel('x'); ylabel('m');
% axis([xlm xlm]);
% legend('MAP','BLS','location','northeastoutside');
% 
