close all;
clear;
clc;

ori_gray = imread('0.tif');  

% figure,  
% imshow(ori_im),  
% title('the original image')  
% ori_gray = rgb2gray(ori_im);  
  
fx = [5 0 -5;8 0 -8;5 0 -5];          % 高斯函数一阶微分，x方向(用于改进的Harris角点提取算法)   
% fx = [-2 -1 0 1 2];                     % x方向梯度算子(用于Harris角点提取算法)   
Ix = filter2(fx,ori_gray);                % x方向滤波   
fy = [5 8 5;0 0 0;-5 -8 -5];          % 高斯函数一阶微分，y方向(用于改进的Harris角点提取算法)   
% fy = [-2;-1;0;1;2];                     % y方向梯度算子(用于Harris角点提取算法)   
Iy = filter2(fy,ori_gray);                % y方向滤波   
Ix2 = Ix.^2;   
Iy2 = Iy.^2;   
Ixy = Ix.*Iy;   
clear Ix;   
clear Iy;   
  
%% 考虑到图像一般情况下的噪声影响，采用高斯滤波去除噪声点  
  
h= fspecial('gaussian',[7 7],2);      % 产生7*7的高斯窗函数，sigma=2   
  
Ix2 = filter2(h,Ix2);   
Iy2 = filter2(h,Iy2);   
Ixy = filter2(h,Ixy);   
height = size(ori_gray,1);   
width = size(ori_gray,2);   
result = zeros(height,width);         % 纪录角点位置，角点处值为1   
  
%% 计算角点的响应函数R（即用一个值来衡量这个点是否是角点）,并标记角点（R(i,j)>0.01*Rmax,且R(i,j)为3x3邻域局部最大值）  
k=1;  
lambda=zeros(height*width,2);  
R = zeros(height,width);   
for i = 1:height   
    for j = 1:width   
        M = [Ix2(i,j) Ixy(i,j);Ixy(i,j) Iy2(i,j)];             % auto correlation matrix   
        K = det(M);                          %求行列式  
        H = trace(M);                        %求迹  
%         R(i,j) = det(M)-0.06*(trace(M))^2;     
        R(i,j) = 4*K/(H^2);  
        lambda(k,:)=[K H];  
        k=k+1;  
    end
end
figure,  
plot(lambda(:,1),lambda(:,2),'.');  
ylabel('trace');xlabel('det');  
  
  
%%  
cnt = 0;   
for i = 2:height-1   
    for j = 2:width-1   
        % 进行非极大抑制，窗口大小3*3   
        if  R(i,j) > R(i-1,j-1) && R(i,j) > R(i-1,j) && R(i,j) > R(i-1,j+1) && R(i,j) > R(i,j-1) && R(i,j) > R(i,j+1) && R(i,j) > R(i+1,j-1) && R(i,j) > R(i+1,j) && R(i,j) > R(i+1,j+1)   
            result(i,j) = 1;   
            cnt = cnt+1;   
        end
    end
end
Rsort=zeros(cnt,1);   
[posr, posc] = find(result == 1);   
  
for i=1:cnt   
    Rsort(i)=R(posr(i),posc(i));   
end
[Rsort,ix]=sort(Rsort,1);   
Rsort=flipud(Rsort);   
ix=flipud(ix);   
ps=120;   
posr2=zeros(ps,1);   
posc2=zeros(ps,1);   
pos=zeros(ps,1);   
for i=1:ps   
    posr2(i)=posr(ix(i));   
    posc2(i)=posc(ix(i));   
    pos(i)= (posr2(i)-1)*width+posc2(i);  
end
  
  
hold on,  
plot(lambda(pos,1),lambda(pos,2),'r*');  
legend('flat & edges','corners')  
  
figure,  
imshow(ori_gray);   
hold on;   
plot(posc2,posr2,'g.');   