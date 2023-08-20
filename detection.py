import cv2 as cv

img = cv.imread('F:\p3.webp')
# cv.imshow('img', img)

haar_casscade = cv.CascadeClassifier('haar_face.xml')
face_rect =  haar_casscade.detectMultiScale(img , scaleFactor= 1.2 , minNeighbors= 2)
print(len(face_rect))

for (x,y,w,h) in face_rect:
    cv.rectangle(img , (x,y), (x+w ,y+h), 255, 2)

cv.imshow("all images",img)
cv.waitKey(0)

