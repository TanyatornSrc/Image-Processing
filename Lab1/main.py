import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("Source/Sample.jpg")

Gimg = cv.cvtColor(image , cv.COLOR_BGR2RGB)
cv.imwrite("Output/Gimg.jpg" , Gimg)

# Convert BGR to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imwrite("Output/Gray.jpg" , gray)

# Histrogram
hist = cv.calcHist([image], [0], None, [255], [0,255])

plt.plot(hist)
plt.title("Red Histogram")
plt.xlabel("Bins")
plt.ylabel("Number of Pixels")
plt.savefig("Output/Hist.jpg")

# Binary image
(thresh, im_bw) = cv.threshold(gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
# cv.imwrite("Output/Bimg_thresh.jpg", thresh)
cv.imwrite("Output/Bimg_bw.jpg", im_bw)

# Resize image
image = cv.resize(image, (250,175))
cv.imwrite("Output/Resize.jpg", image)

# Rotate image
rotate = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
cv.imwrite("Output/Rotate.jpg", rotate)

# GaussianBlur Filter
gfilter = cv.GaussianBlur(image, (9, 9), 0)
cv.imwrite("Output/GaussianBlur_Filter.jpg", gfilter)

# MedianBlur Filter
mfilter = cv.medianBlur(image, 9)
cv.imwrite("Output/MedianBlur_Filter.jpg", mfilter)