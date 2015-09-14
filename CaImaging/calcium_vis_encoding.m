% ENCODING MODEL OF DIRECTION
% Calcium Imaging Data Analysis
% Neuro Bootcamp 2015

% AUTHORS: Vy Vo & Tommy Sprague

% TODO: change this to YOUR path where you've stored the data
root = 'C:/Users/vav001/Google Drive/UCSD Neuro/bootcamp2015/organized_data';

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

% first pass - raw amplitude during sitm period = "response"
all_stim_response = squeeze(mean(all_traces(:,1:stim_nsamples,:),2));
all_stim_speed = squeeze(mean(all_speed(:,1:stim_nsamples),2));

% identify excitatory and inhib cells
inhib_cells = find(vis(1).inh)';
excit_cells = find(~vis(1).inh)';

which_cells = [excit_cells inhib_cells]; % these are what we'll plot

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
% TODO: fill in the values for the stim mask & plot this using imagesc

% We will then multiply this stimulus mask by the basis set.
% This gives us *predicted responses in stimulus space* for each trial.
trnX = stim_mask*basis_set;

% TODO: what is the rank of the design matrix trnX? If it's less than full
% rank (e.g., rank < n_dir_chans) you should NOT proceed, because there are
% too few stimuli for the number of basis functions we have specified.

% TODO: plot the predicted stimulus response for a few different trials.
% TODO: examine the design matrix using imagesc.

%%

% Now that we've setup the basis functions & the design matrix, we can
% train our encoding model using part of the data, and test using the
% remaining part. Just as with classification, we do not want these
% datasets to overlap, but we can repeat this analysis for several
% iterations or folds to get an idea of the average model performance.

% TODO: figure out how many trials you want to train on.
% trial_prctile = ;    % use this % of trials

% Sometimes it helps to only use data that contributes a lot to the model.
% In this case, we can use just a certain percentage of cells that are very
% direction-selective & throw out the rest. You can change this rule to see
% what happens.
% If you load your direction selectivity index, you can use the cells with > median DSI
% usecells = find(all_dsi > median(all_dsi));

% TODO: figure out how many iterations you want to use.
% nfolds = ;
chan_resp = nan([nfolds floor(size(trnX,1)*(1-trial_prctile)), size(trnX,2)]);

trnIdx = []; tstIdx = []; trn = []; tst = [];
for rr = 1:nfolds
    
    % TODO: pick a random subset of trials for training and testing
    trnIdx = ;
    tstIdx = ;
    trn = ;     % should be trials x cells
    tst = ;     % should be trials x cells
    
    % TODO: now train the model using our design matrix & the training
    % data. Essentially you're just regressing the design matrix against
    % the data. (e.g., WeightMatrix = DesignMatrix \ Data
    
    w = ;
    
    % TODO: now invert the weight matrix & apply it to the test data.
    % You need to use the pseudoinverse here since your matrix is likely
    % not perfectly invertible.
    % this will look like: (inv(w*w')*w*testdata')
    
end

% Now plot your mean direction reconstruction.
% TODO: coregister all of your reconstructions across trials so that you
% can average across all of the reconstructions.
% TODO: find the mean direction tuning function, and calculate the standard
% error across your validation folds.