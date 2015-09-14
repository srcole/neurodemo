% fMRI visual orientation decoding
% Neuro Bootcamp 2015

% AUTHOR: Scott Cole

%% 0. Load data
subj = {'s01','s02','s03'};
n_subj = length(subj);
root = strcat(pwd,'\');

attnside = nan(n_subj,n_trials); % Visual hemifield attended. 1=right; 2=left
orL = nan(n_subj,n_trials); % Orientation in left hemifield of each trial
orR = nan(n_subj,n_trials); % Orientation in right hemifield of each trial
betaL = cell(n_subj,1); % Average z-scored activation for each left hemisphere voxel during each trial
betaR = cell(n_subj,1); % Average z-scored activation for each right hemisphere voxel during each trial
nLvox = nan(n_subj,1); % Number of voxels in left hemisphere V1
nRvox = nan(n_subj,1); % Number of voxels in right hemisphere V1
for ss = 1:n_subj
    fn = sprintf('%s%s_data.mat',root,subj{ss});
    load(fn);
    attnside(ss,:) = myAttnSide;
    orL(ss,:) = myOrL;
    orR(ss,:) = myOrR;
    betaL{ss} = myLbetas;
    betaR{ss} = myRbetas;
    nLvox(ss) = size(myLbetas,2);
    nRvox(ss) = size(myRbetas,2);
    clear myAttnSide myLbetas myOrL myOrR myRbetas;
end

n_oris = max(orL(1,:)); % # of unique grating orientations
n_trials = size(betaL{1},1); % # of trials per subject
trialslist = 1:n_trials;
rng(0);

%% 1. Define decoding parameters (left hemisphere only)

% Choose a subject (1, 2, or 3)
ss = 1;

% Choose the fraction of trials to be in the test set
testfrac = .3;
testN = round(n_trials*testfrac);

% Choose the number of surrogate tests
nRuns = 100;


%% 2. Perform decoding with random shuffling

% Iterate through decoding algorithm for the number of surrogate tests
realacc = zeros(nRuns,1);
shufacc = zeros(nRuns,1);
for surr = 1:nRuns
    
    % Generate a random list of trials for the testing and training sets
    testset = NaN;
    trainset = NaN;
    
    % Define the labels for the testing and training sets
    trainlab = NaN;
    testlab = NaN;
    
    % Perform decoding
    % HINT: Use the 'classify.m' function with 
    pred = classify(testset,trainset,trainlab,'diaglinear');
    
    % Calculate accuracy of decoder
    realacc(surr) = NaN;
    
    % Perform decoding and calculate accuracy with shuffled data
    predshuf = classify(testset,trainset,trainlab(randperm(length(trainlab))),'diaglinear');
    shufacc(surr) = NaN;
end


%% 3. Compare the distribution of classifier accuracies between shuffled and non-shuffled decoding



%% 4. Plot a confusion matrix


