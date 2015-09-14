% TUNING FUNCTIONS
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

%% compute orientation selectivity index (OSI) & the direction selectivity index (DSI) through time

% To compute direction selectivity, take the normalized difference between
% the response to the preferred direction & the response to the opposite
% direction. Use the aligned traces generated above.
% DSI: mu_pref - mu_opp / mu_pref + mu_opp

% TODO: compute DSI.

% To compute orientation selectivity, take the normalized difference 
% between the response to the preferred direction & the mean response 
% to the orthogonal directions.
% OSI: mu_pref - mean(mu_orthog) / mu_pref + mean(mu_orthog)
% Note: in orientation space, you actually have two orthogonal orientations
% for every preferred orientation. e.g., for a preferred direction of 180,
% the orthogonal orientations are 90 and 270. That's why you need to take
% the mean across those two (e.g., mean(mu_orthog))

% TODO: compute OSI.

% What should be the bounds of these selectivity indices?

% TODO: Now compute OSI & DSI on the mean responses (rather than on the
% trace through time).

plot_osi_dsi = 1;
if plot_osi_dsi == 1
   
    figure;
    
    % TODO: plot the OSI & DSI over time
    
    % TODO: plot histograms for the mean response OSIs / DSIs
    
    % Did you find outliers in this data? What did they look like?

end

%% How does running speed affect the recorded calcium signal?

% An easy way to look at this is to do a median split: plot the mean
% responses of the cells while the animal is running at speeds lower than
% the median, and at speeds higher than the median.

% TODO: determine if calcium signals or tuning functions differ with
% running speed.