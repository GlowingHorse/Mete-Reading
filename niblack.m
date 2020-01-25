%NIBLACK local thresholding.
function output = niblack(image, varargin)
% Initialization
% clear;
close all;
numvarargs = length(varargin);
% only want 4 optional inputs at most
if numvarargs > 4
    error('myfuns:somefun2Alt:TooManyInputs', ...
     'Possible parameters are: (image, [m n], k, offset, padding)');
end
 
optargs = {[3 3] -0.2 0 'replicate'};
% set defaults
 
optargs(1:numvarargs) = varargin;
% use memorable variable names
[window, k, offset, padding] = optargs{:};

if ~ismatrix(image)
    error('The input image must be a two-dimensional array.');
end

image = double(image);

mean = averagefilter(image, window, padding);
%get from exchange for matlab website

% Standard deviation
meanSquare = averagefilter(image.^2, window, padding);
deviation = (meanSquare - mean.^2).^0.5;
%get the image mean and deviation value for all the image through default
%window size.

output = zeros(size(image));
v = max(k * deviation, offset);
% v = k * deviation;
% Niblack
output(image > mean - v) =0;
output(image <= mean - v) = 1;
%There is just like the slide show.


clc;
