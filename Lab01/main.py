import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("Source/img.jpg")

image = cv.cvtColor(image , cv.COLOR_BGR2RGB)

# Split image layer
# r , g , b = cv.split(image)
r = image[: , : , 0]
g = image[: , : , 1]
b = image[: , : , 2]

# Cal histrogram
r_hist = cv.calcHist([r] , [0] , None , [255] , [0,255])
g_hist = cv.calcHist([g] , [0] , None , [255] , [0,255])
b_hist = cv.calcHist([b] , [0] , None , [255] , [0,255])

# Define figure
plt.subplots(nrows = 4 , ncols = 2, figsize=(8, 8))   

# Position 1 
plt.subplot(4 , 2 , 1)
plt.imshow(image)

# Position 2
plt.subplot(4 , 2 , 2)
plt.plot(r_hist , color = "red")
plt.plot(g_hist , color = "green")
plt.plot(b_hist , color = "blue")

# Position 3
plt.subplot(4 , 2 , 3)
plt.imshow(r, cmap="Reds")
plt.title("Red")

# Position 4
plt.subplot(4 , 2 , 4)
plt.plot(r_hist , color = "red")
plt.title("Red Histogram")

# Position 5
plt.subplot(4 , 2 , 5)
plt.imshow(g, cmap="Greens")
plt.title("Green")

# Position 6
plt.subplot(4 , 2 , 6)
plt.plot(g_hist , color = "green")
plt.title("Green Histogram")

# Position 7
plt.subplot(4 , 2 , 7)
plt.imshow(b, cmap="Blues")
plt.title("Blue")

# Position 8
plt.subplot(4 , 2 , 8)
plt.plot(b_hist , color = "blue")
plt.title("Blue Histogram")

plt.tight_layout()
plt.show()
