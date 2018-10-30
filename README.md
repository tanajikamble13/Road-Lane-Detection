# Road-Lane-Detection

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.

The pipeline goes as follow: 

1 - Convert the image to gray scale
2 - Apply Gaussian blurr to the image to smoothen the image
3 - Apply canny edge detection algorithm
4 - Apply a filter to remove the unwanted area in the image, like the one above the horizon
5 - Apply hough transform and plot the lines that are formed from the points deteccted in the canny edge step.
6 - Based on the slope find the left lines and right lines
7 - Find the largest left and right lines.
8 - Consider an imaginary horizontal line in the middle of image and another line at the bottom of the image.
9 - Find the intersect of largest left and right lines on these imaginary lines using the cramers rule. Plot a line with these intersect point.
