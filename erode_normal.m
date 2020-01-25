%this is a program for common matlab's erosion and dilation method.
%it's slower than the ICM method, when using in open operation.
close all;clc;
clear;
image=imread('0.tif');
I=connection(image);
I(I==1)=255;

% I(I~=255)=0;
% I = double(I);

tic;
se = strel('disk',9);
%set a structure for erosion and dilation.
for i=1:2
    erodedBW = imerode(I,se);
end

for i=1:2
    IM2 = imdilate(erodedBW,se);
end

averageTime_erode_normal = toc/1000.;
%averageTime_erode_normal is the run-time

image=imread('original_img.tif');
% displayImg = rgb2gray(displayImg);
figure('NumberTitle', 'off', 'Name', 'image');
imshow(image,[]);
% figure('NumberTitle', 'off', 'Name', 'This is displayImg');
% imshow(displayImg,[]);
% image(image==255)=0;

image2 = IM2 + double(image);
figure('NumberTitle', 'off', 'Name', 'This is displayImg');
imshow(image2,[]);
clear;
