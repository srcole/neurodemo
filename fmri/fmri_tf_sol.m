% fMRI visual orientation tuning functions SOLUTIONS
% Neuro Bootcamp 2015

% AUTHOR: Scott Cole & Tommy Sprague

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
align_to = 4; % align to 80 degree orientation bin
ori_prefL = cell(n_subj,1);
ori_prefR = cell(n_subj,1);
vtfsLa = cell(n_subj,1);
vtfsRa = cell(n_subj,1);
for ss = 1:n_subj
    % Calculate preferred orientation for each voxel
    [~,ori_prefL{ss}] = max(vtfsL{ss},[],2);
    [~,ori_prefR{ss}] = max(vtfsR{ss},[],2);
    
    % Align tuning curves to the same direction
    vtfsLa{ss} = nan(size(vtfsL{ss}));
    for vv = 1:nLvox(ss)
        vtfsLa{ss}(vv,:) = circshift(vtfsL{ss}(vv,:),[0 align_to-ori_prefL{ss}(vv)]);
    end
    
    vtfsRa{ss} = nan(size(vtfsR{ss}));
    for vv = 1:nRvox(ss)
        vtfsRa{ss}(vv,:) = circshift(vtfsR{ss}(vv,:),[0 align_to-ori_prefR{ss}(vv)]);
    end
end

figure
for ss = 1:n_subj
    subplot(2,n_subj,ss)
    plot(mean(vtfsLa{ss},1));
    ylabel('Average beta, Left')
    xlabel('Orientation')
    title(strcat('Subject ',num2str(ss)))
    
    subplot(2,n_subj,ss+n_subj)
    plot(mean(vtfsRa{ss},1));
    ylabel('Average beta, Right')
    xlabel('Orientation')
end

%% 3. Plot the distribution of preferred orientations for each subject and hemisphere
oris = 0:20:160;
figure()
for ss = 1:n_subj
    subplot(1,n_subj,ss)
    hL = hist(ori_prefL{ss},n_oris);
    hR = hist(ori_prefR{ss},n_oris);
    plot(oris,hL); hold on
    plot(oris,hR)
    legend({'left','right'})
    ylabel('# voxels')
    xlabel('preferred orientation')
    title(strcat('Subject ',num2str(ss)))
end

%% 4. Compute, plot, and compare VTFs for attended vs. nonattended

vtfsLA = cell(n_subj,1);
vtfsRA = cell(n_subj,1);
vtfsLN = cell(n_subj,1);
vtfsRN = cell(n_subj,1);
vtfsLaA = cell(n_subj,1);
vtfsRaA = cell(n_subj,1);
vtfsLaN = cell(n_subj,1);
vtfsRaN = cell(n_subj,1);

for ss = 1:n_subj
    % left hemisphere
    vtfsLA{ss} = nan(nLvox(ss),n_oris);
    vtfsLN{ss} = nan(nLvox(ss),n_oris);
    for ii = 1:nLvox(ss)
        for oo = 1:n_oris
            vtfsLA{ss}(ii,oo) = mean(betaL{ss}(orL(ss,:)==oo & attnside(ss,:)==2, ii),1);
            vtfsLN{ss}(ii,oo) = mean(betaL{ss}(orL(ss,:)==oo & attnside(ss,:)==1, ii),1);
        end
    end
    
    % right hemisphere
    vtfsRA{ss} = nan(nRvox(ss),n_oris);
    vtfsRN{ss} = nan(nRvox(ss),n_oris);
    for ii = 1:nRvox(ss)
        for oo = 1:n_oris
            vtfsRA{ss}(ii,oo) = mean(betaR{ss}(orR(ss,:)==oo & attnside(ss,:)==1, ii),1);
            vtfsRN{ss}(ii,oo) = mean(betaR{ss}(orR(ss,:)==oo & attnside(ss,:)==2, ii),1);
        end
    end
    
    % Align tuning curves to the same direction
    vtfsLaA{ss} = nan(size(vtfsLA{ss}));
    vtfsLaN{ss} = nan(size(vtfsLA{ss}));
    for vv = 1:nLvox(ss)
        vtfsLaA{ss}(vv,:) = circshift(vtfsLA{ss}(vv,:),[0 align_to-ori_prefL{ss}(vv)]);
        vtfsLaN{ss}(vv,:) = circshift(vtfsLN{ss}(vv,:),[0 align_to-ori_prefL{ss}(vv)]);
    end
    
    vtfsRaA{ss} = nan(size(vtfsRA{ss}));
    vtfsRaN{ss} = nan(size(vtfsRA{ss}));
    for vv = 1:nRvox(ss)
        vtfsRaA{ss}(vv,:) = circshift(vtfsRA{ss}(vv,:),[0 align_to-ori_prefR{ss}(vv)]);
        vtfsRaN{ss}(vv,:) = circshift(vtfsRN{ss}(vv,:),[0 align_to-ori_prefR{ss}(vv)]);
    end
end

figure
for ss = 1:n_subj
    subplot(2,n_subj,ss)
    plot(mean(vtfsLaA{ss},1)); hold on
    plot(mean(vtfsLaN{ss},1)); hold on
    plot(mean(vtfsLa{ss},1));
    legend({'attend','not attend','average'})
    ylabel('Average beta, Left')
    xlabel('Orientation')
    title(strcat('Subject ',num2str(ss)))
    
    subplot(2,n_subj,ss+n_subj)
    plot(mean(vtfsRaA{ss},1)); hold on
    plot(mean(vtfsRaN{ss},1)); hold on
    plot(mean(vtfsRa{ss},1));
    legend({'attend','not attend','average'})
    ylabel('Average beta, Right')
    xlabel('Orientation')
end
