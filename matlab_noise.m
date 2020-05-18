clear;clc;
addpath FP_matlab FP_matlab/Filter
dirName = 'iPhone_7p';
imageNoiseSigma = 5;

for i = 1:2
    result_path = ['./matlab_noise/' dirName '/' num2str(i)];
    if ~exist(result_path,'dir') 
        mkdir(result_path);
    end 
    
    ext = {'*.jpg','*.png','*.bmp','*.JPG'};
    img_path   =  [];
    for e=1:length(ext)
        img_path = cat(1,img_path,dir([dirName '/' num2str(i) '/' ext{e}]));
    end
    for j = 1:length(img_path)
        Images(1).name=fullfile(img_path(j).folder,img_path(j).name);

        RP = getFingerprint(Images,imageNoiseSigma);
        RP = rgb2gray1(RP);
        sigmaRP = std2(RP);
        Noisex = WienerInDFT(RP,sigmaRP);
        
        splitname = split(img_path(j).name,'.');
        save(fullfile(result_path, [splitname{1},'.mat'] ),'Noisex')
    end
end