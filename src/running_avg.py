import cv2
import numpy as np
 
c = cv2.VideoCapture('m1.webm')
_,f = c.read()

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (480,360))

avg1 = np.float32(f)
avg2 = np.float32(f)
 
while(1):
    _,f = c.read()
    #cv2.accumulateWeighted(f,avg1,0.1)
    cv2.accumulateWeighted(f,avg2,0.01)
     
    #res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)
 
    #cv2.imshow('img',f)
    #cv2.imshow('avg1',res1)
    #cv2.imshow('avg2',res2)
    res3 = cv2.flip(res2,0)
    out.write(res2)

    k = cv2.waitKey(20)
 
    if k == 27:
        break
 
cv2.destroyAllWindows()
out.release()
c.release()
