function fill_bw=connection(image)
% image=imread('original_img.tif');
% imshow(niblack(imread('eight.tif'), [25 25], -0.2, 10));
output = niblack(image, [9 9],0.2,6,'replicate');
% imwrite(output,'myclown.png')
%get the bwImg
labeled_bw = bwlabel(output);
stats=regionprops(labeled_bw, 'Area');
%get the Area property for get the max-area region (the scale region)

Area_stats = cat(1, stats.Area);
ind = find(Area_stats ==max(Area_stats));
I1=zeros(size(image));
I1(labeled_bw == ind) = 1;
fill_bw = imfill(I1,'holes');
figure('NumberTitle', 'off', 'Name', 'Output');
aaa = ~fill_bw;
imshow(aaa,[]);
print(gcf,'-djpeg','abc.jpg')
%return the scale region for the ICM to do a open operation.

