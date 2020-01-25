close all;
clear;
clc;

ori_gray = imread('0.tif');  

% figure,  
% imshow(ori_im),  
% title('the original image')  
% ori_gray = rgb2gray(ori_im);  
  
fx = [5 0 -5;8 0 -8;5 0 -5];          % ��˹����һ��΢�֣�x����(���ڸĽ���Harris�ǵ���ȡ�㷨)   
% fx = [-2 -1 0 1 2];                     % x�����ݶ�����(����Harris�ǵ���ȡ�㷨)   
Ix = filter2(fx,ori_gray);                % x�����˲�   
fy = [5 8 5;0 0 0;-5 -8 -5];          % ��˹����һ��΢�֣�y����(���ڸĽ���Harris�ǵ���ȡ�㷨)   
% fy = [-2;-1;0;1;2];                     % y�����ݶ�����(����Harris�ǵ���ȡ�㷨)   
Iy = filter2(fy,ori_gray);                % y�����˲�   
Ix2 = Ix.^2;   
Iy2 = Iy.^2;   
Ixy = Ix.*Iy;   
clear Ix;   
clear Iy;   
  
%% ���ǵ�ͼ��һ������µ�����Ӱ�죬���ø�˹�˲�ȥ��������  
  
h= fspecial('gaussian',[7 7],2);      % ����7*7�ĸ�˹��������sigma=2   
  
Ix2 = filter2(h,Ix2);   
Iy2 = filter2(h,Iy2);   
Ixy = filter2(h,Ixy);   
height = size(ori_gray,1);   
width = size(ori_gray,2);   
result = zeros(height,width);         % ��¼�ǵ�λ�ã��ǵ㴦ֵΪ1   
  
%% ����ǵ����Ӧ����R������һ��ֵ������������Ƿ��ǽǵ㣩,����ǽǵ㣨R(i,j)>0.01*Rmax,��R(i,j)Ϊ3x3����ֲ����ֵ��  
k=1;  
lambda=zeros(height*width,2);  
R = zeros(height,width);   
for i = 1:height   
    for j = 1:width   
        M = [Ix2(i,j) Ixy(i,j);Ixy(i,j) Iy2(i,j)];             % auto correlation matrix   
        K = det(M);                          %������ʽ  
        H = trace(M);                        %��  
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
        % ���зǼ������ƣ����ڴ�С3*3   
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