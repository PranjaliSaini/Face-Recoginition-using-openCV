import cv2
import numpy as np
cam = cv2.VideoCapture(0);
ret, img = cam.read()
detector=cv2.CascadeClassifier('F:\Downloads\Pranjali\opencv\haarcascades\haarcascade_frontalface_default.xml');

Id=raw_input('enter your id')
sampleNum=0;
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+str(Id) +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    cv2.imshow('face',img)
    cv2.waitKey(2);
    #wait for 100 miliseconds 
    
    # break if the sample number is morethan 20
    if(sampleNum>20):
        break
cam.release()
cv2.destroyAllWindows()
