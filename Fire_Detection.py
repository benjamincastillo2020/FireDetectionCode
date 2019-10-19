
# coding: utf-8

# In[2]:

import cv2
import numpy as np
import skimage
#import serial
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#get_ipython().magic(u'matplotlib inline')


# In[2]:

cap1 = cv2.VideoCapture('test_fire_1.mp4')
cap2 = cv2.VideoCapture('test_fire_2.mp4')
cap3 = cv2.VideoCapture('test_fire_3.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
#serport = serial.Serial("COM1", 115200)

while(1):
    ret, frame = cap2.read() # reading the image
    sub_image = fgbg.apply(frame) #background subtraction
    
    ret,thresh = cv2.threshold(sub_image,127,255,0) #thresholding
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #finding the contours
    
    areas = [cv2.contourArea(c) for c in contours]
    if len(areas) > 1 :
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        if areas[max_index] > 70:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            string_ = "fire" + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h)
#            serport.write(string_)
            print(string_)
            cv2.putText(frame, 'fire', (x, y), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)
            cv2.imshow('fire detection', frame)
        else :
            print("none")
    
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap1.release()
cap2.release()
cap3.release()
cv2.destroyAllWindows()


# In[ ]:



