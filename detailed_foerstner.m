
close all
clear
clc
G = imread('0.tif');

SIGMA_N          = 2;
%阈值滤除选项
DIFF_KERNEL      = 'gaussian2d';
INT_KERNEL       = 'gaussian';
SIGMA_DIFF       = 1.41;
%梯度方向核大小和sigma

SIGMA_INT        = 1.41 * SIGMA_DIFF *5;
% SIGMA_INT_comp        = 1.41 * SIGMA_DIFF *2;
%M矩阵卷积时核大小和sigma
PREC_THRESH      = 0.5;
%和SIGMA_N一起没什么用

ROUNDN_THRESH    = 0.2;
ALPHA_CLASSI     = 0.999;
DETECTION_METHOD = 'foerstner';
VISUALIZATION    = 'on';

radius_NONMAXSUP = ceil( sqrt( 12 * SIGMA_INT^2 + 5) );
%此处不断增大

global g sigma_n

g       = im2double(G);           
sigma_n = SIGMA_N/255;  
[rows, cols, chans] = size(g);

[Diff_r, Diff_c] = ip_fop_diff_kernel(DIFF_KERNEL, SIGMA_DIFF);
%得到图像梯度滤波器

[Int, radius_INT] = ip_fop_int_kernel(INT_KERNEL, SIGMA_INT);
% [Int_comp, radius_INT_comp] = ip_fop_int_kernel(INT_KERNEL, SIGMA_INT_comp);
%用来平滑核的
% ux = imfilter(g,Diff_c);
% uy = imfilter(g,Diff_r);
% uxy = imfilter(ux,Diff_r);
% imshow(uxy, []);
% uxx = imfilter(ux,Diff_c);
% imshow(uxx, []);
% uyy = imfilter(uy,Diff_r);

% uphiphi = ((ux.^2) .* uxx - 2 .* ux .* uy .* uxy + (uy .^2) .* uyy )/((ux.^2) + (uy .^2));
% imshow(uphiphi, []);

r=1; c=2;
dg(:,:,r)=imfilter(g,Diff_r);
dg(:,:,c)=imfilter(g,Diff_c);
%dg梯度图像,Diff_r检测y方向,Diff_c检测x方向

% clear Diff_r Diff_c g

rr=1; cc=2; rc=3;
gamma(:,:,rr:cc) = dg(:,:,r:c) .^2;
gamma(:,:,rc)    = dg(:,:,r) .* dg(:,:,c);
% clear dg


mean_gamma = imfilter(gamma,Int);
% mean_gamma_comp = imfilter(gamma,Int_comp);
% figure
% subplot(1,2,1);
% imshow(mean_gamma(:,:,1),[]);
% imshow(mean_gamma(:,:,2),[]);
% imshow(mean_gamma(:,:,3),[]);
% print -djpeg -r300 Ix2guassian
% subplot(1,2,2);
% imshow(mean_gamma_comp(:,:,3),[]);
%mean_gamma为高斯卷积后的
 
Spur = mean_gamma(:,:,rr) + mean_gamma(:,:,cc);
% Spur = inv(mean_gamma(:,:,rr)) + inv(mean_gamma(:,:,cc));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%M矩阵的迹等于特征值之和，也就是每一个子M矩阵的特征值之和
Det  = mean_gamma(:,:,rr) .* mean_gamma(:,:,cc) - mean_gamma(:,:,rc).^2;
%M矩阵的行列式等于特征值之积

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w = Det ./ (Spur + eps);
% w = Spur + eps;
%计算特征值的调和平均数，2/(1/lambda1 + 1/lambda2)=2lambda1*lambda2/(lambda1+lambda2)
%正如halcon所写这个是inhomo测各向异性的
% imshow(w,[]);

% 关键滤除阈值
Tw=(sigma_n/PREC_THRESH)^2;  

q = 4 * Det ./ ( Spur.^2 + eps) ;
% imshow(q,[]);
%print -djpeg -r300 isotropy
%算法自带公式q直观来看就是测圆度的
%这个公式测量各向同性并且可以测量圆度
%4(lambda1 * lamda2)/(lambda1 + lambda).^2 =q
%这个lambda就是椭圆的长短轴长短轴相等时为圆，所以可以测圆度
% clear Spur Det

% figure
% subplot(1,2,1);
% imshow(w,[]);
% subplot(1,2,2);
% imshow(q,[]);

candidate_regions = ((q>ROUNDN_THRESH).*(w>Tw)).*w;     
%给特征值的调和平均数加mask
% clear q

% 设计极大值抑制方法选框大小
maxregionMask = imregionalmax( candidate_regions );
weighted_window_centers    =  maxregionMask.* w;
% clear candidate_regions w    
[rWin,cWin] = find(weighted_window_centers > 0);

if length(rWin) == (rows*cols)
    rWin=[];
    cWin=[];
end

win=[];

nonmax_window = -radius_NONMAXSUP:radius_NONMAXSUP;
border  = max([radius_INT, radius_NONMAXSUP])+1;
upperborder_r = rows-border;
upperborder_c = cols-border;

idx = find((rWin>border) .* (cWin>border) .* (rWin<upperborder_r) .* (cWin<upperborder_c));
rWin = rWin(idx);
cWin = cWin(idx);
%去除太靠边的点

%利用极大值抑制不断消除非极值区域
for i=1:length(rWin)
    rW = rWin(i);
    cW = cWin(i);    
    compWin = weighted_window_centers( rW + nonmax_window, cW + nonmax_window);        
    if (weighted_window_centers(rW, cW) >= max(max(compWin))) 
        cent.r=rW;
        cent.c=cW;
        win =[win, cent];
    end    
end
clear weighted_window_centers rWin cWin

%别人的极值点提取方法
radius_EST = ceil( sqrt( 12 * SIGMA_INT ^ 2 + 1 ) / 2 ) + 1;

N_EST      = ( 2 * radius_EST + 1 ) ^ 2;

redundancy_EST = N_EST - 2;

corner     = [];
dot_point  = [];
noclass    = [];

[dc,dr]    = meshgrid(-radius_EST:radius_EST,-radius_EST:radius_EST);  
      
for k=1:length(win)   

    rw = win(k).r;
    cw = win(k).c;
    %取出win保存的点也就是窗口中心点

    ri  = rw + dr;
    ci  = cw + dc;
    %ri为在刚刚取出的点附近生成的meshgrid网格
                 
    idx_r      = rw - radius_EST : rw + radius_EST;  
    idx_c      = cw - radius_EST : cw + radius_EST; 
    %和ri长得很像但是idx_r为单行
                        
    gamma_rr = gamma(idx_r,idx_c,rr);
    gamma_cc = gamma(idx_r,idx_c,cc);
    gamma_rc = gamma(idx_r,idx_c,rc);                
    %gamma是被高斯核滤波前的M
    %从gamma中找到这些点附近的点

    sum_gr_gr   = sum(sum(gamma_rr));
    sum_gc_gc   = sum(sum(gamma_cc));
    sum_gr_gc   = sum(sum(gamma_rc)); 
    %sum_gr_gr这里是gamma具体值，估计用来做权重之类的
    
    sum_gr_gr_r = sum(sum( gamma_rr .* ri ));
    sum_gr_gr_c = sum(sum( gamma_rr .* ci ));            
    sum_gc_gc_r = sum(sum( gamma_cc .* ri ));
    sum_gc_gc_c = sum(sum( gamma_cc .* ci ));            
    sum_gr_gc_r = sum(sum( gamma_rc .* ri ));
    sum_gr_gc_c = sum(sum( gamma_rc .* ci ));
    %这里则是利用gamma_rr加权后的点的具体坐标，ri是之前的meshgrid网格

    corner_N     = [  sum_gr_gr ,  sum_gr_gc ;  sum_gr_gc, sum_gc_gc ];
    corner_l     = [ sum_gr_gr_r + sum_gr_gc_c;  sum_gr_gc_r + sum_gc_gc_c];
    %这为什么选择这几个加权点应该有具体数学算法说明
    
    corner_N_inv = inv( corner_N );

    corner_est   = corner_N\corner_l;
    %此处已经求得corner点坐标
    corner_residuals_r  = ri - corner_est(1);
    corner_residuals_c  = ci - corner_est(2);                        

    corner_residuals_rr = corner_residuals_r .^ 2;
    corner_residuals_rc = corner_residuals_r .* corner_residuals_c;
    corner_residuals_cc = corner_residuals_c .^ 2;

    corner_Omega = sum(sum( gamma_rr .* corner_residuals_rr + 2 * gamma_rc .* corner_residuals_rc + gamma_cc .* corner_residuals_cc ));

    tmp            = corner_est';
    new_point.r    = tmp(1);
    new_point.c    = tmp(2);

    new_point.cov  = corner_Omega/redundancy_EST * corner_N_inv;                     
    new_point.cov  = new_point.cov;

    corner = [corner, new_point];               
                      
    tmp = ([win(k).r; win(k).c])';
    win(k).r = tmp(1);
    win(k).c = tmp(2);     
end

% Visualization
if strcmp(VISUALIZATION,'on')
    figure
    imshow(G)
    hold on        
    % Fenstermitten     
    % Visualisierung : r und c vertauschen.            
    plot([win.c], [win.r], 'r+', 'MarkerSize', 10);                
    % Fenster:         
    for t=1:length(win)                    
        rw=win(t).c;       
        cw=win(t).r;
        line([rw-radius_EST, rw+radius_EST],[cw-radius_EST,cw-radius_EST],'Color','red');           
        line([rw-radius_EST, rw+radius_EST],[cw+radius_EST,cw+radius_EST],'Color','red');           
        line([rw-radius_EST, rw-radius_EST],[cw-radius_EST,cw+radius_EST],'Color','red');           
        line([rw+radius_EST, rw+radius_EST],[cw-radius_EST,cw+radius_EST],'Color','red');                       
    end    
        % Falls vorhanden, Eckpunkte anzeigen
    if ~isempty(corner)    
        plot([corner.c],[corner.r],'bx', 'MarkerSize',8);    

        % Fehlerellipsen plotten
        for i=1:length(corner)                
            %% Man BEACHTE den FAKTOR 100...
            [xell,yell]=ip_errell(corner(i).r, corner(i).c, 100*corner(i).cov);            
            plot(yell, xell,'b-');
        end
    end    
    hold off
end   

disp('Results:');
% a) integer positions of window centers
disp('Positions of window centers (actually were integers in the internally scaled image):');
for i=1:length(win)
    fprintf('%5.1f   %5.1f\n',win(i).r,win(i).c);
end

% b) subpixel positions of corners with covariance matrix
disp('Subpixel positions of corners with covariance matrix');
for i=1:length(corner)
    r = corner(i).r;
    c = corner(i).c;
    cov = corner(i).cov;
end

function [Diff_r, Diff_c] = ip_fop_diff_kernel(DIFF_KERNEL, SIGMA_DIFF)

alpha_fractile = 3.3;
switch DIFF_KERNEL
    case 'gaussian1d'
        radius = ceil(alpha_fractile*SIGMA_DIFF);
        x      = -radius:radius;
        
        Diff_c=1/(sqrt(2*pi) * SIGMA_DIFF^3) * x .* exp(- x.^2 ./ (2 * SIGMA_DIFF^2) );    
        Diff_r=Diff_c';
    case 'gaussian2d'
        radius = ceil(alpha_fractile*SIGMA_DIFF);
        x      = -radius:radius;

        Diff_c = 1/(sqrt(2*pi) * SIGMA_DIFF^3) * x .* exp(- x.^2 ./ (2 * SIGMA_DIFF^2) ); 
        gaussian_kernel = fspecial('gaussian',[length(x) 1],SIGMA_DIFF);
        Diff_c = conv2(Diff_c, gaussian_kernel);
%         aaa = gaussian_kernel(2)*Diff_c;%这就是此处卷积的含义
        Diff_r=Diff_c';    
end

end

function [Int, radius_INT] = ip_fop_int_kernel(INT_KERNEL, SIGMA_INT)

alpha_fractile = 3.3;
switch INT_KERNEL
    case 'gaussian'     
        radius_INT = ceil( alpha_fractile * SIGMA_INT );
        Int        = fspecial('gaussian', 2 * radius_INT + 1 , SIGMA_INT);  

end

end