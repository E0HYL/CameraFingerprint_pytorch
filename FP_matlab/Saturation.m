function SaturMap = Saturation(X,gray)
% Determines saturated pixels as those having a peak value (must be over
% 250) and a neighboring pixel of equal value
% SaturMap  binary matrix, 0 - saturated pixels 
% gray      in case of 'gray' saturated pixels in SaturMap (denoted as zeros)
%           result from at least 2 saturated color channels

if nargin<2, gray=''; end

M = size(X,1);  N = size(X,2);
if max(max(X(:)))<=250; 
    if isempty(gray), 
        SaturMap = ones(size(X)); 
    else 
        SaturMap = ones(M,N); 
    end
    return, 
end

Xh = (X - circshift(X,[0,1]));  
Xv = (X - circshift(X,[1,0]));
SaturMap = Xh & Xv & circshift(Xh,[0,-1]) & circshift(Xv,[-1,0]);

if size(X,3)==3, 
    for j=1:3
        maxX(j) = max(max(X(:,:,j)));
        if maxX(j)>250; 
            SaturMap(:,:,j) = ~((X(:,:,j)==maxX(j)) & ~SaturMap(:,:,j));
        end
    end
elseif size(X,3)==1,
    maxX = max(max(X)); 
    SaturMap = ~((X==maxX) & ~SaturMap);    
else error('Invalid matrix dimensions)');
end

switch gray
    case 'gray'
        if size(X,3)==3,
            SaturMap = SaturMap(:,:,1)+SaturMap(:,:,3)+SaturMap(:,:,3);
            SaturMap(SaturMap>1) = 1;
        end
    otherwise 'do nothing';
end