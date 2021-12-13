import cv2

# detect the cascade, to use it later
trained_face_data = cv2.CascadeClassifier('./Resources/haarcascade_frontalface_default.xml')

# testing with a preloaded image
img = cv2.imread("./Resources/Images/ironman4.jpg")

# change image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# place gray image in algo
# detectMultiScale = detects all the faces, no matter if faces are bigger or smaller.
face_coordinates = trained_face_data.detectMultiScale(imgGray)
# show coordinates of face in the image
print(face_coordinates)

# draw rectangles and loop it!
# each detected face gets a rectangle
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Dude", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
# (x, y, w, h) = face_coordinates[0]




# show image
cv2.imshow("Final image", img)
# pauses the termination of the program
cv2.waitKey(0)


print("Sve ok")
