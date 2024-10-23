import cv2

image = cv2.imread('test.jpg')

# No.1
cv2.rectangle(image, (50, 50), (200, 200), (0, 0, 255), 2)
cv2.imshow('display', image)

#---------------------------------------------------------------------

# No.2
# cropped = image[100:400, 95:405]
# cv2.imshow('cropped', cropped)
# cv2.imshow('display', image)

#---------------------------------------------------------------------

# No.3
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow('display', thresh)

#---------------------------------------------------------------------

# No.4
# font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(image, 'Hello Informatics', (117,25), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
# cv2.imshow('display', image)

#---------------------------------------------------------------------

# No.5
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# equ = cv2.equalizeHist(gray)
# cv2.imshow('display', gray)
# cv2.imshow('contrast', equ)

#---------------------------------------------------------------------

# No.6
# import cv2

# image = cv2.imread('test.jpg')

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# for (x, y, w, h) in faces:
#     cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)  # วาดกรอบใบหน้าด้วยสีฟ้า
#     roi_gray = gray[y:y+h, x:x+w]  # ส่วนของใบหน้าในภาพขาวดำ
#     roi_color = image[y:y+h, x:x+w]  # ส่วนของใบหน้าในภาพสี

#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  # วาดกรอบดวงตาด้วยสีเขียว

# cv2.imshow('display', image)

#---------------------------------------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()