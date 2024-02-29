import cv2


fullbody_cascade=cv2.CascadeClassifier("haarcascade_fullbody.xml")


# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fullbody = fullbody_cascade.detectMultiScale(gray,1.1,5)
    #Convert Each Frame into Grayscale
    
    # Pass frame to our body classifier
    
    for(x,y,w,h) in fullbody:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('frame', frame)
      
    # Extract bounding boxes for any bodies identified
    

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cap.release()
cv2.destroyAllWindows()
