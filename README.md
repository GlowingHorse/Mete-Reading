# meter_reading
The code for the paper of "Auto-reading method for precision pointer meter based on ICM"

## Introduction
A matlab implementation of the method proposed in the paper "Auto-reading method for precision pointer meter based on ICM".
We try to make the variables in the codes as consistent as possible in the paper.

## How to use
1. We have provide a test image here, you can download the code and test it directly.
2. Some description for these scripts:
* The "ICM_get_best_iter" is for computing the iteration times.
* The "ICM_use_best_iter" is for using the iteration computed before to erode and dilate the image.
* The "erode_normal" is for testing the Matlab operators about image erosion and dilation.
* The "forstner_harris_compare" is a file containing Forstner's and Harris's point detection algorithms for further comparison.
* The "detailed_foerstner" is a detailed description of the Forstner algorithm.

## Try it
1. Clone or download it to read these codes and try to run them.
