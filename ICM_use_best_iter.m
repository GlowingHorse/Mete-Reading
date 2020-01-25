clear;
image=imread('original_img86.tif');
Best_iter=8;
%this Best_iter is calculated in ICM_get_best_iter.m
fill_bw=connection(image);
fill_bw=~fill_bw;
S = fill_bw;
I = fill_bw;
%Get I

tic;
Y=zeros(size(I));
F=Y;
Theta=Y+1.4999;
A_g=0.7;
A_h=5;
A_f=0.9;

K=[1 1 1;1 1 1;1 1 1];
M=zeros(size(I));
for i=1:Best_iter
  W=conv2(Y,K,'same');
%   F=F*A_f +double(S)+ W;
  F=F*A_f +double(S)+ W;
  M=M+Y;
  Y=double(F>Theta);
  
%   figure('NumberTitle', 'off', 'Name', '这是F');
%   imshow(F,[]);
%   figure('NumberTitle', 'off', 'Name', '这是M');
%   imshow(M,[]);
%   figure('NumberTitle', 'off', 'Name', '这是Y');
%   imshow(Y,[]);
  
  Theta=A_g*Theta + A_h*Y;
  close all;
  %还原显示
  if(min(min(M))>0)
    iteration_times =i;
    break;
  end
end
%All above is like do a erosion operation
S2 = ~Y;

Y2=zeros(size(S2));
F2=Y2;
Theta2=Y2+1.4999;
A_g2=0.7;
A_h2=5;
A_f2=0.9;

for i=1:Best_iter
  W2=conv2(Y2,K,'same');
  F2=F2*A_f2 +double(S2)+ W2;
  Y2=double(F2>Theta2);
  
%   figure('NumberTitle', 'off', 'Name', '这是F2');
%   imshow(F2,[]);
%   figure('NumberTitle', 'off', 'Name', '这是Y2');
%   imshow(Y2,[]);
  
  Theta2=A_g2*Theta2 + A_h2*Y2;
  close all;
  %还原显示
%   if(min(M(:))>0)
%     k=i;
%     break;
%   end
end
%All above there like do a dilation operation

% 
labeled_bw = bwlabel(Y2);
stats=regionprops(labeled_bw, 'Area');
Area_stats = cat(1, stats.Area);
ind = find(Area_stats ==max(Area_stats));

averageTime_ICM = toc/1000.;
%get the run-time to compare with file erode_normal.m

% displayImg = rgb2gray(displayImg);
% figure('NumberTitle', 'off', 'Name', '这是displayImg');
% imshow(displayImg,[]);
% displayImg(displayImg==255)=0;

% figure('NumberTitle', 'off', 'Name', 'This is displayImg');
subplot(1,2,1);
imshow(image,[]);
title('image');
I1=zeros(size(Y2));
I1(labeled_bw == ind) = 1;
displayImg = 255*I1 + double(image);

% figure('NumberTitle', 'off', 'Name', 'This is I1');
subplot(1,2,2);
imshow(displayImg,[]);
title('displayImg');
clear;
