% DECODING DIRECTION
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
trnPct = ;
tstPct = ;
nFolds = ;

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
%     trn = ;
%     tst = ;
%     
%     trnlabel = ;
%     tstlabel = ;
    
    % We use a function from the Statistics toolbox to do the
    % classification.
    % type 'doc classify' into the Command Window for more information on
    % this function.
    class = classify(tst,trn,trnlabel,'diaglinear');
    % There are many other types of classifiers we can use, but given the
    % limited number of trials from this dataset we need to stick to the
    % naive Bayes classifier (e.g., the 'diaglinear' option).
    
    % TODO: calculate the accuracy of the classifier!
%     all_acc(nf) = ;
    
    % fill confusion matrix
    for cc = 1:length(class)
        conf_mat(class(cc),tstg(cc)) = conf_mat(class(cc),tstlabel(cc))+1;
    end

end

% TODO: plot your confusion matrix using imagesc.

% BONUS: Can you figure out if this decoder is statistically significant?
% The best way to do this is to calculate classification accuracy for a
% completely random set of labels on the data on every fold/iteration of
% your classification analysis.
% Then you compare the classifier performance for the real data & the
% shuffled data.