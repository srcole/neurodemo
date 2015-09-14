% fMRI visual orientation tuning functions
% Neuro Bootcamp 2015

% AUTHORS: Scott Cole & Tommy Sprague

%% 0. Load data
subj = {'s01','s02','s03'};
n_trials = 288; % # of trials per subject
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

%% 1. Compute VTFs (visual tuning functions) for each voxel
vtfsL = cell(n_subj,1);
vtfsR = cell(n_subj,1);
for ss = 1:length(subj)
    % left hemisphere
    vtfsL{ss} = nan(nLvox(ss),n_oris);
    for ii = 1:nLvox(ss)
        for oo = 1:n_oris
            vtfsL{ss}(ii,oo) = mean(betaL{ss}(orL(ss,:)==oo, ii),1);
        end
    end
    
    % right hemisphere
    vtfsR{ss} = nan(nRvox(ss),n_oris);
    for ii = 1:nRvox(ss)
        for oo = 1:n_oris
            vtfsR{ss}(ii,oo) = mean(betaR{ss}(orR(ss,:)==oo, ii),1);
        end
    end
end

%% 2. Plot an average VTF for each subject and hemisphere
% HINT: align voxels' VTFs so that all have the same preferred orientation


%% 3. Plot the distribution of preferred orientations for each subject and hemisphere


%% 4. Compute, plot, and compare VTFs for attended vs. nonattended

