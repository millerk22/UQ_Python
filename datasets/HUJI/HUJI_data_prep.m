%% HUJI data prep for use in Python code
clear all
clc

ratio =  huji_sampling_60seg(1);           
data_root = './data/HUJI/';         %Location of unshuffled data
temp_root = './data/HUJI/temp/';    % Shuffled data
load([data_root,'ground_truth_huji_disney_60seg.mat'])
load([temp_root, 'train_test_index.mat'], 'train_index', 'test_index');
fprintf("Training set: %2f%%\n", 100 * numel(train_index)/numel(ground_truth))





%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Nystrom 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nystrom_opt             = {};
nystrom_opt.tau         = -2;
nystrom_opt.K = 40;
nystrom_opt.Laplacian   = 'n';
nystrom_opt.Metric      = 'Euclidean';
nystrom_opt.numsample   = 400;
outstr = struct_description(nystrom_opt);
Nystrom_huji(nystrom_opt);
load([data_root,'VE_',outstr,'.mat'],'phi','E')
shuffle_index = [train_index test_index];
phi = phi(shuffle_index,:);
ground_truth = ground_truth(shuffle_index);
fid_opts.type = 'random';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Down sampling?
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ratio           = floor(ratio / 5);
ground_truth    = ground_truth(1:5:length(ground_truth));
phi             = phi(1:5:size(phi,1),:);


%% Save the data
fp = strcat(['../UQ_Python/datasets/HUJI/HUJI_data_',outstr,'.mat']);
save(fp, 'E', 'phi', 'ground_truth', 'ratio');

%%











%%
function Nystrom_huji(nystrom_opt)
global data_root
    outstr = struct_description(nystrom_opt);
    fp = [data_root, 'VE_',outstr,'.mat'];
    if exist(fp,'file')
       fprintf('Found precomputed eigenvectors/values\n');
       return
    end
    load([data_root,'H_huji_disney_60seg.mat'],'H');
    H = movmean(H,20,2);
    disp("Performing Nystrom extension")
    tic;
    [phi, E] = nystrom(H', nystrom_opt);
    t = toc;
    fprintf("Nystrom takes %f seconds\n", t);
    clear H 
    save(fp, 'nystrom_opt', 'phi', 'E');
end
