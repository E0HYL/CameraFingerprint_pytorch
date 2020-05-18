%{
clear;clc;
addpath PCE_matlab
dirName='./matlab_noise/iPhone_7p';

for i=1:2
    fingerprint_path = dir([dirName '/' num2str(i) '/*.mat']);
    for j=1:length(fingerprint_path)
        load([fingerprint_path(j).folder,'/',fingerprint_path(j).name]);
        Fingerprint = rgb2gray1(Noisex);
        [M,N] = size(Fingerprint);
        if(M < N)
            Fingerprint = imrotate(Fingerprint,90);
        end
        Noisex_d{j} = Fingerprint;
    end
    Noise{i} = Noisex_d;
end


Noise_1 = Noise{1};
Noise_2 = Noise{2};
PCE0_t = [];
PCE1_device1_t = [];
PCE1_device2_t = [];

for i=34:size(Noise_1,2)
    if i==34
    for j = 66:size(Noise_2,2)
        C = crosscorr(Noise_1{i},Noise_2{j});
        detection = PCE(C);
        disp(['PCE0 ' num2str(i) ',' num2str(j) ': ' num2str(detection.PCE)])
        PCE0_t = [PCE0_t;[detection.PCE i j]];
    end        
    else
    for j = 1:size(Noise_2,2)
        C = crosscorr(Noise_1{i},Noise_2{j});
        detection = PCE(C);
        disp(['PCE0 ' num2str(i) ',' num2str(j) ': ' num2str(detection.PCE)])
        PCE0_t = [PCE0_t;[detection.PCE i j]];
    end
    end
end
%}
for i=5:size(Noise_1,2)-1
    if i==5
    for j= 38:size(Noise_1,2)
        C = crosscorr(Noise_1{i},Noise_1{j});
        detection = PCE(C);
        disp(['PCE1_1 ' num2str(i) ',' num2str(j) ': ' num2str(detection.PCE)])
        PCE1_device1_t = [PCE1_device1_t;[detection.PCE i j]];        
    end        
    else
    for j= i+1:size(Noise_1,2)
        C = crosscorr(Noise_1{i},Noise_1{j});
        detection = PCE(C);
        disp(['PCE1_1 ' num2str(i) ',' num2str(j) ': ' num2str(detection.PCE)])
        PCE1_device1_t = [PCE1_device1_t;[detection.PCE i j]];        
    end
    end
end

for i=1:size(Noise_2,2)-1
    for j= i+1:size(Noise_2,2)
        C = crosscorr(Noise_2{i},Noise_2{j});
        detection = PCE(C);
        disp(['PCE1_2 ' num2str(i) ',' num2str(j) ': ' num2str(detection.PCE)])
        PCE1_device2_t = [PCE1_device2_t;[detection.PCE i j]];        
    end
end

result_path = [dirName '/result/'];

if ~exist(result_path,'dir') 
    mkdir(result_path);
end 

csvwrite(strcat(result_path,'PCE0.csv'),PCE0_t);
csvwrite(strcat(result_path,'PCE1_device1.csv'),PCE1_device1_t);
csvwrite(strcat(result_path,'PCE1_device2.csv'),PCE1_device2_t);
