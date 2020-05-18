function I_new = preprocess(srcFileName,Resolution)
%CROP 此处显示有关此函数的摘要
%   此处显示详细说明
    I = imread(srcFileName);
    I_size = size(I);
    if I_size(1) > I_size(2)
        I = rot90(I,-1);
    end
%     I_new = I(1:Resolution,1:Resolution,:);
    I_new = imresize(I,[Resolution,Resolution]);
end
