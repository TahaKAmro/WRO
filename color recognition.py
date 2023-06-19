import cv2 as cv
import numpy as np
import imutils
import keyboard

cap=cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

while True:
    
    _, frame = cap.read()
    frame = cv.resize(frame,(1000,750))
    blur = cv.GaussianBlur(frame,(15,15),0)
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    
    lower_green=np.array([40, 70, 80])
    upper_green=np.array([70, 255, 255])

    lower_red=np.array([0, 50, 120])
    upper_red=np.array([10, 255, 255])

#########################################################################################################################   
    mask_red=cv.inRange(hsv,lower_red,upper_red)
    mask_green=cv.inRange(hsv,lower_green,upper_green)

    cnts_green=cv.findContours(mask_green,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cnts_green=imutils.grab_contours(cnts_green)
#########################################################################################################################
    cnts_red=cv.findContours(mask_red,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cnts_red=imutils.grab_contours(cnts_red)
    

#########################################################################################################################
    for c in cnts_green:
                
        area_green=cv.contourArea(c)
            
        if area_green>5000:
            cv.drawContours(frame,[c],-1,(0,255,0),2)
            M=cv.moments(c)
            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])

            cv.circle(frame,(cx,cy),7,(255,255,255),-1)
            cv.putText(frame,"green",(cx-20,cy-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    
    for c in cnts_red:
                
        area_red=cv.contourArea(c)
        if area_red>5000:

            cv.drawContours(frame,[c],-1,(0,0,255),2)
            M=cv.moments(c)

            cx=int(M["m10"]/M["m00"])
            cy=int(M["m01"]/M["m00"])

            cv.circle(frame,(cx,cy),7,(255,255,255),-1)
            cv.putText(frame,"red",(cx-20,cy-20),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

            
    cv.imshow("result",frame)
    k=cv.waitKey(5)
    if k==27:
        break
cap.release()
cv.destroyAllWindows()