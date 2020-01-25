%This code is just for get the best iteration time from run the ICM througn
%all the image, in our example, the meter image need run 30times before all
%the meter region was iterated.
clear;
image=imread('original_img86.tif');
fill_bw=connection(image);
% imshow(fill_bw,[]);
%the connection is function in the connection.m 
fill_bw=~fill_bw;
S = fill_bw;

I = fill_bw;
%Get I as show in slides

tic;
%time starts
Y=zeros(size(I));
F=Y;
Theta=Y+1.4999;

%Y F Theta are all like they are defined in the slides
A_g=0.7;
% A_g=0.9;
A_h=5;
% A_h=11;
A_f=0.9;
% A_f=0.7;
%g,h,f are setten as they were in my paper.

K=[1 1 1;
    1 1 1;
    1 1 1];
%connection matric k

M=zeros(size(I));
%M is the matrix named sequential matrix(the scale image in fifth and sixth pages of slides)
for i=1:30
    W=conv2(Y,K,'same');
  %this conv2 is used the Matlab function, but if we can use GPU the run it
  %, the speed will improve many times. Like "convn(gpuArray.rand(100,100,10,'single');"
    F=F*A_f +double(S)+ W;
    Y=double(F>Theta);
    M=M+Y;
  %All those are like showed in slides
%   fig = figure;
%   fig.Position = get(0,'ScreenSize');
%   zoom on;
%   subplot(1,3,1);
%   imshow(F,[]);
%   title('This is F');
%   subplot(1,3,2);
%   imshow(Theta,[]);
%   title('This is Theta');
%   subplot(1,3,3);
%   imshow(Y,[]);
%   title('This isY');

%     imshow(M,[]);
%     close all;
%   %close all windows

	Theta=A_g*Theta + A_h*Y;
%     imshow(M,[]);
%     imwrite(F, strcat(num2str(i),'Fite.png'));
%     imwrite(M, strcat(num2str(i), 'Mite.png'));
%     imwrite(Theta, strcat(num2str(i), 'Thetaite.png'));
%     imwrite(Y, strcat(num2str(i), 'Yite.png'));
    
%   Theta=A_g*Theta;
    if(min(min(M))>0)
        iteration_times =i;
        break;
    end
  
  %this if determine statements are to get the proper M
  %the M is used to show How the matrix changes with iteration.
end
% 
% I = uint8(I);
% T1 = graythresh(I)*255;
% w = zeros(256,1);
% p = w;
% M = max(I(:));
% imshow(M,[]);

k = max(max(M));
%k is the maximum in M and the total iteration times
tau = round(10/18*k);
%set the parameter for get the optimal iteration times
y_a = zeros(k-1,1);
p = y_a;

x_a(1:1:tau)=1:1:tau;
y_a(1:1:tau)=(1-(x_a./(tau*2)).^2).^2;
x_a(tau+1:1:k)=(tau+1):1:k;
y_a(tau+1:1:k)=(tau./(2*x_a(tau+1:1:k))).^2;
%weighed for Otsu as showed in fixth of slides

% plot(x_a, y_a);
[m,n]=size(M);
for i=1:1:k
    sumup = sum(sum((M==i)));
    %number of grayvalue is i, just like otsu's P(this P is in wiki's meaning)
    p(i)=y_a(i)*sumup/(m*n);
    %in there, the original otsu's P has been changed with the weighed
    %value y_a(i) become a new weighted p(i)
end


p_norm=p/sum(p);
%normalization
sigma=zeros(k-1,1);
%sigma is the list to contain the inter-class variance later
for i=1:k-1
% for i=1:k

    p1=sum(p_norm(1:i));
    p2=sum(p_norm(i+1:k));
%     mu1=dot((1:i),p(1:i))/sum(p(1:i));
    mu1=dot((1:i),p(1:i))/p1;
%     a = (i:k-1);
%     b = p((i+1):k);

%     mu2=dot(((i+1):k),p((i+1):k))/sum(p((i+1):k));
    mu2=dot(((i+1):k),p((i+1):k))/p2;
%     mu2=dot(a,b);
    sigma(i)=p1*p2*(mu1-mu2)^2;
    % To get the inter-class variance like Otsu method
end
[~,iter_before]=max(sigma);
%Using otsu method to find the proper iteration times
Best_iter = k - iter_before;
%The reason use k - iter_before can be found from the image M
%In there M maximum is 25, so when theshold value is 17, we can get the
%scale perfectly, that means that we only need to iterate 25-17 = 8 times
%to get the perfect scale.

%Best_iter is the number means that when the ICM model run Best_iter times
%we can get the scale region and we don't need to run the model with the
%total iteration times
clc;