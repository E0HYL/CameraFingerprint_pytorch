function [RP,LP,ImagesinRP] = getFingerprint(Images,sigma) 
% -------------------------------------------------------------------------
% Copyright (c) 2012 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: mgoljan@binghamton.edu | January 2012
%          http://dde.binghamton.edu/download/camera_fingerprint
% -------------------------------------------------------------------------
% Extracts and averages noise from all Images and outputs a camera fingerprint 
% INPUT:
%    Images      list of color images to process (obtained through dir command)
%                these images have to be from the same camera and the same size
%    sigma       local std of extracted noise
% OUTPUT:
%    RP          reference pattern - estimate of PRNU (in the output file)
%    LP          linear pattern structured data
% EXAMPLE:
% addpath(strcat(pwd,'\Functions'))
% addpath(strcat(pwd,'\Filter'))
%       image_directory = '\TestImages';
%       Images = dir([image_directory,'\*.jpg']);
%       addpath(image_directory)
%       [RP,LP,ImagesinRP] = getFingerprint(Images);
% -------------------------------------------------------------------------
% [1] M. Goljan, T. Filler, and J. Fridrich. Large Scale Test of Sensor 
% Fingerprint Camera Identification. In N.D. Memon and E.J. Delp and P.W. Wong and 
% J. Dittmann, editors, Proc. of SPIE, Electronic Imaging, Media Forensics and 
% Security XI, volume 7254, pages % 0I?01?0I?12, January 2009.
% -------------------------------------------------------------------------

database_size = length(Images);             % Number of the images
if database_size==0, error('No images of specified type in the directory.'); end
if nargin<2, sigma = 3; end                 % local std of extracted noise

%%%  Parameters used in denoising filter
L = 4;                                      %  number of decomposition levels
qmf = MakeONFilter('Daubechies',8);
       
t=0; 
for i=1:database_size
    SeeProgress(i),
    if isstruct(Images)
        im = Images(i).name;
    else
        im = Images{i};
    end
    % X = preprocess(im, 299);
    X = imread(im); 
    X = double255(X);
    if t==0
        [M,N,three]=size(X);
        if three==1 
            continue;                           % only color images will be processed    
        end
        %%%  Initialize sums 
        for j=1:3
            RPsum{j}=zeros(M,N,'single');   
            NN{j}=zeros(M,N,'single');        	% number of additions to each pixel for RPsum
        end
    else
        s = size(X);
        if length(size(X))~=3, 
            fprintf('Not a color image - skipped.\n');
            continue;                           % only color images will be used 
        end
        if any([M,N,three]~=size(X))
            fprintf('\n Skipping image %s of size %d x %d x %d \n',im,s(1),s(2),s(3));
            continue;                           % only same size images will be used 
        end
    end
    % The image will be the t-th image used for the reference pattern RP
    t=t+1;                                      % counter of used images
    ImagesinRP(t).name = im;
    
    for j=1:3
        ImNoise = single(NoiseExtract(X(:,:,j),qmf,sigma,L)); 
        Inten = single(IntenScale(X(:,:,j))).*Saturation(X(:,:,j));    % zeros for saturated pixels
        RPsum{j} = RPsum{j}+ImNoise.*Inten;   	% weighted average of ImNoise (weighted by Inten)
        NN{j} = NN{j} + Inten.^2;
    end
    
end

clear ImNoise Inten X
if t==0, error('None of the images was color image in landscape orientation.'), end
RP = cat(3,RPsum{1}./(NN{1}+1),RPsum{2}./(NN{2}+1),RPsum{3}./(NN{3}+1));
% Remove linear pattern and keep its parameters
[RP,LP] = ZeroMeanTotal(RP);
RP = single(RP);               % reduce double to single precision          

    
%%% FUNCTIONS %%
function X=double255(X)
% convert to double ranging from 0 to 255
datatype = class(X);
    switch datatype,                % convert to [0,255]
        case 'uint8',  X = double(X);
        case 'uint16', X = double(X)/65535*255;  
    end
