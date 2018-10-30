#!/usr/bin/env python
# coding: utf-8

# # **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note** If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# In[1]:


#importing some useful packages
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image


# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


import math

def line_fun(p1,p2):
    a = (p1[1] - p2[1])
    b = (p2[0] - p1[0])
    c = (p2[0]*p1[1] - p1[0]*p2[1])
    return a,b,c
  
def intersect(l1,l2):
    D = l1[0]*l2[1] -l1[1]*l2[0]  
    Dx = l1[2]*l2[1] -l1[1]*l2[2]
    Dy = l1[0]*l2[2] -l1[2]*l2[0]
    if D != 0:
        x = (Dx)/D
        y = (Dy)/D
        return int(x),int(y)
    else:
        x = -1
        y = -1
        return int(x),int(y)


def draw_lines(img, lines, color=[0, 0, 255], thickness=12):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating li
    ne segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    left_lar_size = 0
    right_lar_size = 0
    right_lar = (0,0,0,0)
    left_lar = (0,0,0,0)
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness) 
            size = math.hypot(x2 - x1, y2 - y1)
            slope = ((y2-y1)/(x2-x1))
            if (slope > 0.5): #right
                if (size > right_lar_size):
                    right_lar = (x1, y1, x2, y2)                    
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness) #All right lines
            
            elif (slope < -0.551): #left
                if (size > left_lar_size):
                    left_lar = (x1, y1, x2, y2)
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness) #All left lines

    #Right Largest
    #cv2.line(img, (right_lar[0], right_lar[1]), (right_lar[2], right_lar[3]), (0,255,0), thickness)

    #Left Largest
    #cv2.line(img, (left_lar[0], left_lar[1]), (left_lar[2], left_lar[3]), (0,255,0), thickness)

#    print ("Start of draw")
#    print (left_lar)
#    print (right_lar)
#    print ("DONE PRINTING LAR")
    imgHeight, imgWidth = (img.shape[0], img.shape[1])
    #UP_LINE
    UP = line_fun([0, int(0.6*imgHeight)],[int(imgWidth), int(0.6*imgHeight)])
#    print ("UP")
#    print (UP)
    #DOWN_LINE
    downLinePoint1 = [0, int(imgHeight)]
    downLinePoint2 = [int(imgWidth), int(imgHeight)]
    DOWN = line_fun(downLinePoint1,downLinePoint2)
#    print ("DOWN")
#    print (DOWN)
    #LEFT LINE
    pl_1 = [left_lar[0],left_lar[1]]
    pl_2 = [left_lar[2],left_lar[3]]
    LEFT_LAR = line_fun(pl_1,pl_2)
#    print ("LEFT_LAR")
#    print (LEFT_LAR)
    #RIGHT
    pr_1 = [right_lar[0],right_lar[1]]
    pr_2 = [right_lar[2],right_lar[3]]
    RIGHT_LAR = line_fun(pr_1,pr_2)
#    print ("RIGHT_LAR")
#    print (RIGHT_LAR)
    #LEFT_LINE INTERSECT POINT
    upLeftPoint = intersect(UP,LEFT_LAR)
    downLeftPoint = intersect(DOWN,LEFT_LAR)
    
    #RIGHT_LINE INTERSECT POINT
    upRightPoint = intersect(UP,RIGHT_LAR)
    downRightPoint = intersect(DOWN,RIGHT_LAR)
    if (upLeftPoint[1] == -1 or downLeftPoint[1] == -1):
        pass 
    else:
        cv2.line(img, (upLeftPoint[0], upLeftPoint[1]), (downLeftPoint[0], downLeftPoint[1]), [255, 0, 0], thickness) #draw left line
    if (upRightPoint[1] == -1 or downRightPoint[1] == -1):
        pass 
    else:
        cv2.line(img, (upRightPoint[0], upRightPoint[1]), (downRightPoint[0], downRightPoint[1]), [255, 0, 0], thickness) #draw left line 
    

    #FOR LEFT LINE DRAW
#    if math.isinf(float(upLeftPoint)) or math.isinf(float(downLeftPoint)):
#        print ("The man who knew infinity")
#        return 
#    else:
        #Put the draw line here
#    print (upLeftPoint + "\n" + downLeftPoint) 
    
    #FOR RIGHT LINE DRAW
#    if math.isinf(float(upRightPoint)) or math.isinf(float(downRightPoint)):
#        print ("The man who knew infinity")
#        return 
#    else:
#        #Put the draw line here
#    print (upLeftPoint + "\n" + downLeftPoint) 

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #draw_lines(line_img, lines)
    return line_img,lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# ## Test on Images
# 
# Now you should build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


# run your solution on all test_images and make copies into the test_images directory).

# In[5]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

image_dir = os.listdir(os.getcwd() + "/test_images/")
for image in image_dir:
    img1 = mpimg.imread('test_images/' + image)
    gray_img = grayscale(img1)
    gauss_img = gaussian_blur(gray_img,5)
    canny_img = canny(gauss_img,50,150)
    imshape = canny_img.shape
    vertices = np.array([[(0,imshape[0]),(.45*imshape[1], 0.6*imshape[0]), (.55*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    mask_img = region_of_interest(canny_img,vertices)
    line_img,lines = hough_lines(mask_img,1,np.pi/180,15,40,20)
    blank_image = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
    #color_edges = np.dstack((canny_img,canny_img,canny_img)) 
    #img_lines = weighted_img(line_img,color_edges)
    draw_lines(blank_image,lines)
    img1 = weighted_img(img1,blank_image)
    cv2.imwrite('test_images/org_img1' + image, cv2.cvtColor(img1,cv2.COLOR_BGR2RGB))

#plt.show()
#grayscale(os.getcwd() + "/test_images/" + image)


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`

# In[7]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[8]:


def process_image(img1):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    gray_img = grayscale(img1)
    gauss_img = gaussian_blur(gray_img,5)
    canny_img = canny(gauss_img,50,150)
    imshape = canny_img.shape
    vertices = np.array([[(0,imshape[0]),(.45*imshape[1], 0.6*imshape[0]), (.55*imshape[1], 0.6*imshape[0]), (imshape[1],imshape[0])]], dtype=np.int32)
    mask_img = region_of_interest(canny_img,vertices)
    line_img,lines = hough_lines(mask_img,1,np.pi/180,50,100,160)
    blank_image = np.zeros((img1.shape[0],img1.shape[1],3), np.uint8)
    #color_edges = np.dstack((canny_img,canny_img,canny_img)) 
    #img_lines = weighted_img(line_img,color_edges)
    draw_lines(blank_image,lines)
    img1 = weighted_img(img1,blank_image)
    return img1[:,:,[0,1,2]]


# Let's try the one with the solid white lane on the right first ...

# In[9]:


white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[10]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# **At this point, if you were successful you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform.  Modify your draw_lines function accordingly and try re-running your pipeline.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[11]:


yellow_output = 'yellow.mp4'
clip2 = VideoFileClip('solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[12]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Reflections
# 
# Congratulations on finding the lane lines!  As the final step in this project, we would like you to share your thoughts on your lane finding pipeline... specifically, how could you imagine making your algorithm better / more robust?  Where will your current algorithm be likely to fail?
# 
# Please add your thoughts below,  and if you're up for making your pipeline more robust, be sure to scroll down and check out the optional challenge video below!
# 
# The pipeline goes as follow:
# 1 - Convert the image to gray scale
# 
# 2 - Apply Gaussian blurr to the image to smoothen the image
# 
# 3 - Apply canny edge detection algorithm
# 
# 4 - Apply a filter to remove the unwanted area in the image, like the one above the horizon
# 
# 5 - Apply hough transform and plot the lines that are formed from the points deteccted in the canny edge step.
# 
# 6 - Based on the slope find the left lines and right lines
# 
# 7 - Find the largest left and right lines.
# 
# 8 - Consider an imaginary horizontal line in the middle of image and another line at the bottom of the image.
# 
# 9 - Find the intersect of largest left and right lines on these imaginary lines using the cramers rule. Plot a line     with these intersect point.
# 
# I have a lot of jittery lines drawn on the video frames. A solutoon to this is to store the values of previous lines and take a average of these lines and plot them accordingly.
# 

# ## Submission
# 
# If you're satisfied with your video outputs it's time to submit!  Submit this ipython notebook for review.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[13]:


challenge_output = 'extra.mp4'
clip2 = VideoFileClip('challenge.mp4')
challenge_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[14]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))


# In[ ]:





# In[ ]:




