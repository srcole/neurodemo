% Calcium Imaging Data Analysis SOLUTIONS
% Neuro Bootcamp 2015

% AUTHORS: Vy Vo & Tommy Sprague

% TODO: change this to YOUR path where you've stored the data
root = strcat(pwd,'\');

%% define some variables based on the data notes

stim_dur = 4; % s
isi_dur  = 8; % s
trial_dur = stim_dur+isi_dur;

directions = ([1:12]-1)*30;

frame_rate = 6.3;           % s
frame_per  = 1/frame_rate;  % s

stim_nsamples = ceil( stim_dur*frame_rate );
isi_nsamples = ceil( isi_dur*frame_rate );
trial_nsamples = stim_nsamples + isi_nsamples;

% also convert this to time so we can plot with appropriate axes
trial_t = (0:(trial_nsamples-1)) * frame_per;
stim_t = (0:(stim_nsamples-1)) * frame_per;
isi_t  = (0:(isi_nsamples -1)) * frame_per;

% load file:
load(fullfile(root, 'ca_data.mat'),'vis');

% we are just going to use the data from one animal (e.g., vis(1).)
% you can replace everything with vis(2) if you want to look at data from
% the second animal.

% let's pull out the data into matrices to make it easier to work with.
tempvisdat = {vis(1).data{:}}';
ntrials = numel(tempvisdat);
ncells = size(tempvisdat{1},1);
% all calcium traces are stored in a matrix with cell x time x trial
all_traces = permute(reshape(cell2mat(tempvisdat),ncells,ntrials,[]),[1 3 2]);
all_stim_labels = vis(1).trialorder(:);
all_speed = cell2mat({vis(1).runspeed{:}}');
clear tempvisdat


% Convert all traces to zscore
dothis = 0;
if dothis
    cellmean = zeros(ncells,1);
    cellstd = zeros(ncells,1);
    for cc = 1:ncells
        cellmean(cc) = mean(reshape(all_traces(cc,:,:),ntrials*size(all_traces,2),1));
        cellstd(cc) = std(reshape(all_traces(cc,:,:),ntrials*size(all_traces,2),1));
    end
    
    all_traces(cc,:,:) = (all_traces(cc,:,:) - cellmean(cc)) / cellstd(cc);
end
    
% first pass - raw amplitude during stim period = "response"
all_stim_response = squeeze(mean(all_traces(:,1:stim_nsamples,:),2));
all_stim_speed = squeeze(mean(all_speed(:,1:stim_nsamples),2));

% identify excitatory and inhib cells
inhib_cells = find(vis(1).inh)';
excit_cells = find(~vis(1).inh)';

which_cells = [excit_cells inhib_cells]; % these are what we'll plot

%% plot TFs for each cell, color-coded by excit/inhib

% TODO: calculate a mean tuning function for each cell for each direction.
% This will require that you take the mean response of the cell to each
% presented direction. You will also want to calculate a standard error of
% the tuning function (which is std(data) ./ sqrt(n) ).
% We have initialized the variables for you below.

all_tfs_m = nan(ncells,length(directions));     % mean tuning function
all_tfs_e = nan(ncells,length(directions));     % std err of tuning function
% also plot the tuning functions through trial time
all_tfs_trace_m = nan(ncells,trial_nsamples,length(directions));

for ii = 1:length(directions)
    all_tfs_m(:,ii) = nanmean(all_stim_response(:,all_stim_labels(:,1)==ii),2);
    all_tfs_trace_m(:,:,ii) = nanmean(all_traces(:,:,all_stim_labels(:,1)==ii),3);
    
    all_tfs_e(:,ii) = nanstd(all_stim_response(:,all_stim_labels(:,1)==ii),[],2)./sqrt(ntrials);
end


%% sort into preferred direction (the direction with the maximum mean response)

% Now we align all the tuning functions to a central orientation/direction
% using circshift.

[~,pref_dir] = max(all_tfs_m,[],2);     % take the max of the 2nd dimension
% align the max to the middle orientation bin
align_to = 6;
all_tfs_aligned_m = nan(size(all_tfs_m));
all_tfs_aligned_e = nan(size(all_tfs_e));
all_tfs_aligned_trace_m = nan(size(all_tfs_trace_m));

for cc = 1:size(all_tfs_m,1)
    all_tfs_aligned_m(cc,:) = circshift(all_tfs_m(cc,:),[0 align_to - pref_dir(cc)]);
    all_tfs_aligned_e(cc,:) = circshift(all_tfs_e(cc,:),[0 align_to - pref_dir(cc)]);
    
    all_tfs_aligned_trace_m(cc,:,:) = circshift(all_tfs_trace_m(cc,:,:),[0 0 align_to-pref_dir(cc)]);
end

% TODO: now plot the following from your aligned tuning function data:
% (1) tuning functions for all excitatory cells, and their mean TF
% (2) tuning functions for all inhibitory cells, and their mean TF
% (3) mean TFs over time (from all_tfs_aligned_trace_m) for excitatory
% cells (hint: use imagesc so that you can display time on the x axis &
% direction selectivity on the y-axis)
% (4) mean TFs over time for inhibitory cells


plot_aligned_tfs = 1;
if plot_aligned_tfs == 1
    
    figure;
    subplot(2,2,1);hold on;
    plot(directions-directions(align_to),all_tfs_aligned_m(excit_cells,:)','-','color',[100 180 100]./255);
    plot(directions-directions(align_to),median(all_tfs_aligned_m(excit_cells,:),1)','-','color',[0 130 0]./255,'linewidth',3);
    ylim([-0.5 3]);xlabel('Direction (offset from preferred)');
    ylabel('Mean stimulus response (dF/F)');
    title('Excitatory cells');
    
    hold off;
    subplot(2,2,2);hold on;
    plot(directions-directions(align_to),all_tfs_aligned_m(inhib_cells,:)','-','color',[200 100 100]./255);
    plot(directions-directions(align_to),median(all_tfs_aligned_m(inhib_cells,:),1)','-','color',[150 0 0]./255,'linewidth',3);
    ylim([-0.5 3]);xlabel('Direction (offset from preferred)');
    ylabel('Mean stimulus response (dF/F)');
    title('Inhibitory cells');
    hold off;
    
    %subplot(2,2,3);imagesc(trial_t,directions-directions(align_to),squeeze( mean( all_tfs_aligned_trace_m(excit_cells,:,:) ,1))');set(gca,'CLim',[-0.1 0.5]); colormap hot;
    subplot(2,2,3);imagesc(trial_t,directions-directions(align_to),squeeze( median( all_tfs_aligned_trace_m(excit_cells,:,:) ,1))');set(gca,'CLim',[-0.1 0.5]); colormap hot;
    xlabel('Time (s)');ylabel('Direction (relative to preferred)');title(sprintf('All excitatory cells (n = %i)',length(excit_cells)));
    %subplot(2,2,4);imagesc(trial_t,directions-directions(align_to),squeeze( mean( all_tfs_aligned_trace_m(inhib_cells,:,:) ,1))');set(gca,'CLim',[-0.1 0.5]); colormap hot;
    subplot(2,2,4);imagesc(trial_t,directions-directions(align_to),squeeze( median( all_tfs_aligned_trace_m(inhib_cells,:,:) ,1))');set(gca,'CLim',[-0.1 0.5]); colormap hot;
    xlabel('Time (s)');ylabel('Direction (relative to preferred)');title(sprintf('All inhibitory cells (n = %i)',length(inhib_cells)));
end

%% compute orientation selectivity index (OSI) & the direction selectivity index (DSI) through time

% To compute direction selectivity, take the normalized difference between
% the response to the preferred direction & the response to the opposite
% direction. Use the aligned traces generated above.
% DSI: mu_pref - mu_opp / mu_pref + mu_opp

% TODO: compute DSI.
all_dsi_trace = (all_tfs_aligned_trace_m(:,:,align_to) - ...
    all_tfs_aligned_trace_m(:,:,align_to+6)) ./ ...
    (all_tfs_aligned_trace_m(:,:,align_to) + ...
    all_tfs_aligned_trace_m(:,:,align_to+6));

% To compute orientation selectivity, take the normalized difference 
% between the response to the preferred direction & the mean response 
% to the orthogonal directions.
% OSI: mu_pref - mean(mu_orthog) / mu_pref + mean(mu_orthog)
% Note: in orientation space, you actually have two orthogonal orientations
% for every preferred orientation. e.g., for a preferred direction of 180,
% the orthogonal orientations are 90 and 270. That's why you need to take
% the mean across those two (e.g., mean(mu_orthog))

% TODO: compute OSI.
orthog = mean(cat(3,all_tfs_aligned_trace_m(:,:,align_to + 3),...
    all_tfs_aligned_trace_m(:,:,align_to - 3)),3);
all_osi_trace = (all_tfs_aligned_trace_m(:,:,align_to) - orthog) ...
    ./ (all_tfs_aligned_trace_m(:,:,align_to) + orthog);
clear orthog;

% TODO: Now compute OSI & DSI on the mean responses (rather than on the
% trace through time).
all_dsi = (all_tfs_aligned_m(:,align_to) - all_tfs_aligned_m(:,align_to+6)) ./ ...
    (all_tfs_aligned_m(:,align_to) + all_tfs_aligned_m(:,align_to+6));
orthog = mean(cat(2,all_tfs_aligned_m(:,align_to+3),all_tfs_aligned_m(:,align_to-3)),2);
all_osi = (all_tfs_aligned_m(:,align_to) - orthog) ./ (all_tfs_aligned_m(:,align_to) + orthog);
clear orthog;

plot_osi_dsi = 1;
if plot_osi_dsi == 1
   
    figure;
    
    subplot(2,2,1);hold on;
    plot(trial_t,median(all_osi_trace(excit_cells,:),1),'g-');
    plot(trial_t,median(all_osi_trace(inhib_cells,:),1),'r-');
    xlabel('Time (s)'); ylabel('OSI');legend({'Excitatory','Inhibitory'});%ylim([-1 1]);
    title('Orientation selectivity');
    hold off;
    
    subplot(2,2,2);hold on;
    plot(trial_t,median(all_dsi_trace(excit_cells,:),1),'g-');
    plot(trial_t,median(all_dsi_trace(inhib_cells,:),1),'r-');
    xlabel('Time (s)'); ylabel('DSI');%ylim([-1 1]);
    title('Direction selectivity');
    hold off;
    
    subplot(2,2,3);hold on;
    hist(all_osi);
    ylabel('OSI');
    
    subplot(2,2,4);hold on;
    hist(all_dsi);
    ylabel('DSI');

end


%% How does running speed affect the recorded calcium signal?

% An easy way to look at this is to do a median split: plot the mean
% responses of the cells while the animal is running at speeds lower than
% the median, and at speeds higher than the median.

% TODO: determine if calcium signals or tuning functions differ with
% running speed.

mean_speed_stim = mean(all_speed(:,1:stim_nsamples),2);
median_mean_speed = median(mean_speed_stim(:));

mean_speed_sorted_responses = nan(length(which_cells),2); % n_cells x above/below median

mean_speed_sorted_responses(:,1) = mean(all_stim_response(which_cells,mean_speed_stim<=median_mean_speed),2);%squeeze(mean(all_traces(which_cells,1:stim_nsamples,mean_speed_stim <= median_mean_speed),2));
mean_speed_sorted_responses(:,2) = mean(all_stim_response(which_cells,mean_speed_stim> median_mean_speed),2);

plot_mean_speed_sorted = 1;
if plot_mean_speed_sorted == 1
    figure;
    plot(mean_speed_sorted_responses(excit_cells,:)','go-'); hold on;
    plot(mean_speed_sorted_responses(inhib_cells,:)','ro-');
    title('Mean calcium responses for a median split of running speed');
    ylabel('Normalized df/f');
end

%% let's try our hand at decoding using a linear classifier.

% You can decide to use all the cells (default here) or just excitatory or
% inhibitory cells.

% which_cells_dec = excit_cells;
% which_cells_dec = inhib_cells;
which_cells_dec = which_cells;

% We'll want to repeat our classification analysis several times (e.g.,
% nFolds times), using a different portion of the data as the training set
% each time. You can play with any of the following parameters, but they
% should not make a huge impact on the results unless you skew them
% enormously (e.g., only 2 folds, or using only 20% of the data to train
% the classifier).
trnPct = 0.7;
tstPct = 0.3;
nFolds = 100;

% Initialize some variables
all_acc = nan(nFolds,1);
% We will eventually make a confusion matrix, which will plot the real
% direction on the x axis and the classified direction on the y axis. If
% the decoding is perfect, you should get a single bright diagonal band.
conf_mat = zeros(length(directions),length(directions));

for nf = 1:nFolds;
    
    % use randsample to get a random set of trials for your training data
    trnidx = randsample(ntrials,ntrials*trnPct);
    tstidx = setxor(1:ntrials,trnidx);
    
    % TODO: fill in both the data & the group labels
    trn = all_stim_response(which_cells_dec,trnidx)';
    tst = all_stim_response(which_cells_dec,tstidx)';
    
    trnlabel = all_stim_labels(trnidx);
    tstlabel = all_stim_labels(tstidx);
    
    % We use a function from the Statistics toolbox to do the
    % classification.
    % type 'doc classify' into the Command Window for more information on
    % this function.
    class = classify(tst,trn,trnlabel,'diaglinear');
    % There are many other types of classifiers we can use, but given the
    % limited number of trials from this dataset we need to stick to the
    % naive Bayes classifier (e.g., the 'diaglinear' option).
    
    % TODO: calculate the accuracy of the classifier!
    all_acc(nf) = nanmean(class==tstlabel);;
    
    % fill confusion matrix
    for cc = 1:length(class)
        conf_mat(class(cc),tstlabel(cc)) = conf_mat(class(cc),tstlabel(cc))+1;
    end

end

% TODO: plot your confusion matrix using imagesc.

plot_conf_mat = 1;
if plot_conf_mat == 1
    figure;
    conf_mat = conf_mat./max(conf_mat(:));
    imagesc(conf_mat);ylabel('classified into');xlabel('real value');
    fprintf('classification accuracy (direction): %0.02f\n',nanmean(all_acc));
end

% BONUS: Can you figure out if this decoder is statistically significant?
% The best way to do this is to calculate classification accuracy for a
% completely random set of labels on the data on every fold/iteration of
% your classification analysis.
% Then you compare the classifier performance for the real data & the
% shuffled data.

hich_cells_dec = which_cells;

trnPct = 0.7;
tstPct = 0.3;
nFolds = 100;
all_acc = nan(nFolds,1);
shuf_acc = nan(nFolds,1);

for nf = 1:nFolds;
    
    trnidx = randsample(ntrials,ntrials*trnPct);
    tstidx = setxor(1:ntrials,trnidx);
    
    trn = reshape(permute(all_traces(which_cells_dec,:,trnidx),[3 1 2]),ntrials*trnPct,[]);
    tst = reshape(permute(all_traces(which_cells_dec,:,tstidx),[3 1 2]),ntrials*tstPct,[]);
    
    trng = all_stim_labels(trnidx);
    tstg = all_stim_labels(tstidx);
    
    class = classify(tst,trn,trng,'diaglinear');
    class_shuf = classify(tst,trn,trng(randperm(ntrials*trnPct)),'diaglinear');
    
    all_acc(nf) = nanmean(class==tstg);
    shuf_acc(nf) = nanmean(class_shuf==tstg);

end

pval = 1 - mean(all_acc-shuf_acc>0);

fprintf('classification accuracy (direction): %0.02f\n', nanmean(all_acc));
fprintf('shuffled accuracy: %0.02f\n', nanmean(shuf_acc));
fprintf('2-tailed p value: %0.02f\n', pval);

pval = 1 - mean(all_acc-shuf_acc>0);

fprintf('classification accuracy (direction): %0.02f\n', nanmean(all_acc));
fprintf('shuffled accuracy: %0.02f\n', nanmean(shuf_acc));
fprintf('2-tailed p value: %0.02f\n', pval);

%% Direction encoding model
% As reviewed, the encoding model essentially regresses the data you've
% obtained onto a set of filters that tile the stimulus space. In our case,
% the stimulus space is any presented direction from 0 to 360 degrees.

% First we need to make a bunch of filters, or basis functions, which tile
% the space. If a certain cell is direction selective at 180 degrees & is
% presented with that direction, the filters near 180 degrees will have
% higher weights/activations than the filters far from 180 degrees.

n_dir_chans = 6;

% for now, you can use this filter shape (a raised cosine).
make_basis_function = @(xx,mu) (cosd( (xx-mu)*0.5 ) ).^(n_dir_chans-mod(n_dir_chans,2));

% Let's look at our channels.
xx = linspace(1,360,360);
basis_set = nan(360,n_dir_chans);
chan_center = linspace(360/n_dir_chans,360,n_dir_chans);

for cc = 1:n_dir_chans
    basis_set(:,cc) = make_basis_function(xx,chan_center(cc));
end

figure;
subplot(1,2,1);plot(xx,basis_set);
xlabel('Direction');
ylabel('Filter amplitude');
title('Basis set');
% now let's see how well the channels tile the space by summing them:
subplot(1,2,2);plot(xx,sum(basis_set,2));
% If the channels tiled the space unevenly, you'd see something other than
% a flat line.

% Now generate a trial-by-trial stimulus mask that is in direction space
% (e.g., for a trial where the presented orientation is 180, the mask
% should be 1 at 180 degrees and 0 everywhere else)
stim_mask = zeros(ntrials,numel(xx));
alldirs = (all_stim_labels-1)*30;
% TODO: fill in the values for the stim mask & plot this using imagesc
for tt = 1:ntrials  % loop over trials
    stim_mask(tt, alldirs(tt)+1) = 1;
end
imagesc(stim_mask);

% We will then multiply this stimulus mask by the basis set.
% This gives us *predicted responses in stimulus space* for each trial.
trnX = stim_mask*basis_set;

% TODO: what is the rank of the design matrix trnX? If it's less than full
% rank (e.g., rank < n_dir_chans) you should NOT proceed, because there are
% too few stimuli for the number of basis functions we have specified.
rank(trnX)

% TODO: plot the predicted stimulus response for a few different trials.
% TODO: examine the design matrix using imagesc.


tr_num = 8;
figure;
subplot(1,3,1);hold on;
plot(xx,basis_set);
plot(xx,stim_mask(tr_num,:),'k-');
xlabel('Orientation (\circ)');title(sprintf('trial %i',tr_num));
xlim([0 360]);
subplot(1,3,2);hold on;
plot(chan_center,trnX(tr_num,:),'k-');
for cc = 1:n_dir_chans
    plot(chan_center(cc),trnX(tr_num,cc),'o','MarkerSize',8,'LineWidth',3);
end
xlabel('Channel center (\circ)');title('Predicted channel response');
xlim([-5 360]);

% and look at the design matrix

subplot(1,3,3);hold on;
imagesc(chan_center,1:size(trnX,1),trnX);
title('Design matrix');
xlabel('Channel center (\circ)');ylabel('Trial'); axis tight ij;

%%

% Now that we've setup the basis functions & the design matrix, we can
% train our encoding model using part of the data, and test using the
% remaining part. Just as with classification, we do not want these
% datasets to overlap, but we can repeat this analysis for several
% iterations or folds to get an idea of the average model performance.

% TODO: figure out how many trials you want to train on.
trial_prctile = 0.7;    % use this % of trials

% Sometimes it helps to only use data that contributes a lot to the model.
% In this case, we can use just a certain percentage of cells that are very
% direction-selective & throw out the rest. You can change this rule to see
% what happens.
% only use the cells with > median DSI
% from above:
all_dsi = (all_tfs_aligned_m(:,align_to) - all_tfs_aligned_m(:,align_to+6)) ./ ...
    (all_tfs_aligned_m(:,align_to) + all_tfs_aligned_m(:,align_to+6));
usecells = find(all_dsi > median(all_dsi));

% TODO: figure out how many iterations you want to use.
nfolds = 50;
chan_resp = nan([nfolds floor(size(trnX,1)*(1-trial_prctile)), size(trnX,2)]);

trnIdx = []; tstIdx = []; trn = []; tst = [];
for rr = 1:nfolds
    
    % TODO: pick a random subset of trials for training and testing
    trnIdx = randsample(ntrials,ntrials*trial_prctile);
    tstIdx = setxor(1:ntrials,trnIdx);
    
    trn = all_stim_response(usecells,trnIdx)';
    tst = all_stim_response(usecells,tstIdx)';
    
    % TODO: now train the model using our design matrix & the training
    % data. Essentially you're just regressing the design matrix against
    % the data. (e.g., WeightMatrix = DesignMatrix \ Data
    
    w = trnX(trnIdx,:)\trn;
    
    
    % TODO: now invert the weight matrix & apply it to the test data.
    % You need to use the pseudoinverse here since your matrix is likely
    % not perfectly invertible.
    % this will look like: (inv(w*w')*w*testdata')
    
    chan_resp(rr,:,:) = (inv(w*w')*w*tst').';
    
end

% Now plot your mean direction reconstruction.
% TODO: coregister all of your reconstructions across trials so that you
% can average across all of the reconstructions.


targ_chan = ceil(n_dir_chans/2);

chan_resp_shift = nan(size(chan_resp));
for ii = 1:size(chan_resp,2)
    chan_resp_shift(rr,ii,:) =  circshift(chan_resp(rr,ii,:),targ_chan-find(directions==alldirs(ii)),2);
end

% TODO: find the mean direction tuning function, and calculate the standard
% error across your validation folds.

mean_chan_resp = squeeze(nanmean(nanmean(chan_resp_shift),2));
se_chan_resp = squeeze(nanstd(nanmean(chan_resp_shift),[],2)) ./ sqrt(nfolds);

figure;
hold on;
plot(chan_center, mean_chan_resp ); hold on
plot(chan_center, mean_chan_resp + se_chan_resp, 'b--');
plot(chan_center, mean_chan_resp - se_chan_resp, 'b--'); 
line([chan_center(targ_chan) chan_center(targ_chan)],[0 1],'Color','k');
xlabel('Orientation channel (\circ)');
ylabel('Channel response');
title('Coregistered direction reconstructions');
hold off;