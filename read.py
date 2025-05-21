import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Rescale function
def rescaleFrame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load the image
img = cv.imread('images.jpg')
if img is None:
    print("Error: Image not found.")
    exit()

# Rescale image
img_rescaled = rescaleFrame(img)

# Convert to grayscale
gray = cv.cvtColor(img_rescaled, cv.COLOR_BGR2GRAY)

# finding histogram 
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(gray_hist)
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.xlim([0, 256])
#plt.show()

#colors histogram
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    hist = cv.calcHist([img_rescaled], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.title('Color Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
#plt.show()

# Normalize histogram
gray_hist = cv.calcHist([gray], [0], None, [256], [0, 256])
gray_hist = cv.normalize(gray_hist, gray_hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
plt.plot(gray_hist)
plt.title('Normalized Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Normalized Frequency')
plt.xlim([0, 256])
#plt.show()
#cv.imshow('Normalized Grayscale Histogram', gray_hist)

# Normalize color histograms
for i, color in enumerate(colors):
    hist = cv.calcHist([img_rescaled], [i], None, [256], [0, 256])
    hist = cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
#cv.imshow(f'Normalized {color} Histogram', hist)



# Normalize histogram   
# Apply Canny edge detection
canny = cv.Canny(cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT), 125, 175)

# Find contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# Create blank canvas and draw contours
blank = np.zeros(img_rescaled.shape[:2], dtype='uint8')
cv.drawContours(blank, contours, -1, (255), 1)  # Use 255 (white) for grayscale image

# Show images
#cv.imshow('Original Rescaled Image', img_rescaled)
#cv.imshow('Contours', blank)




    # Grayscale conversion
gray = cv.cvtColor(img_rescaled, cv.COLOR_BGR2GRAY)

    # BGR to HSV
hsv = cv.cvtColor(img_rescaled, cv.COLOR_BGR2HSV)
    # BGR to LAB
lab = cv.cvtColor(img_rescaled, cv.COLOR_BGR2LAB)
    # BGR to LUV
luv = cv.cvtColor(img_rescaled, cv.COLOR_BGR2LUV)
    # BGR to YUV
yuv = cv.cvtColor(img_rescaled, cv.COLOR_BGR2YUV)
    # BGR to YCrCb
ycrcb = cv.cvtColor(img_rescaled, cv.COLOR_BGR2YCrCb)
    # BGR to XYZ
xyz = cv.cvtColor(img_rescaled, cv.COLOR_BGR2XYZ)
    # BGR to HLS
hls = cv.cvtColor(img_rescaled, cv.COLOR_BGR2HLS)
# edge detection
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian', lap)

# Sobel x
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobelx = np.uint8(np.absolute(sobelx)) 
cv.imshow('Sobel X', sobelx)
# Sobel y   
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
sobely = np.uint8(np.absolute(sobely))
cv.imshow('Sobel Y', sobely)
# Sobel combined
sobel_combined = cv.bitwise_or(sobelx, sobely)
cv.imshow('Sobel Combined', sobel_combined)

# canny edge detection
canny = cv.Canny(gray, 125, 175)
cv.imshow('Canny', canny)

    # Blur
blur = cv.GaussianBlur(img_rescaled, (5, 5), cv.BORDER_DEFAULT)

    # Edge detection
canny = cv.Canny(blur, 125, 175)

    # Dilation
dilated = cv.dilate(canny, (7, 7), iterations=3)

    #erosion
eroded = cv.erode(dilated, (7, 7), iterations=3)

    #resize
resized = cv.resize(eroded, (500, 500), interpolation=cv.INTER_CUBIC)

    #find  contours
contours, heirarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(gray, contours, -1, (0, 255, 0), 1)

    # contours using threshold
ret, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    # Inverse threshold
ret, thresh2 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    # Adaptive threshold
adaptive_thresh = cv.adaptiveThreshold(cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    # Otsu's threshold
otsu_thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


# different thresholding methods
#cv.imshow('inv Threshold', thresh2)
#cv.imshow('Adaptive Threshold', adaptive_thresh)
#cv.imshow('Otsu Threshold', otsu_thresh)    
    

# Show image results
#cv.imshow('Original', img_rescaled)
#cv.imshow('Gray', gray)
#cv.imshow('Blurred', blur)
#cv.imshow('Canny', canny)
#cv.imshow('Dilated', dilated)
#cv.imshow('Eroded', eroded)
#cv.imshow('Resized', resized)
#cv.imshow('Contours', gray)
# cv.imshow('hsv', hsv)
# cv.imshow('lab', lab)
# cv.imshow('luv', luv)
# cv.imshow('yuv', yuv)
# cv.imshow('ycrcb', ycrcb)
# cv.imshow('xyz', xyz)
# cv.imshow('hls', hls)
#cv.imshow('Threshold', thresh1)

print(f'Number of contours found: {len(contours)}')
cv.waitKey(0)
cv.destroyAllWindows()

# Process video
#capture = cv.VideoCapture(0)  # Use 0 for webcam

if not capture.isOpened():
    print("Error: Cannot open video file.")
else:
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            print("Reached end of video or failed to read frame.")
            break
        blured = cv.GaussianBlur(frame, (3, 3), cv.BORDER_DEFAULT)
        cannied = cv.Canny(blured, 125, 175)
        dilateded = cv.dilate(cannied, (7, 7), iterations=3)

        frame_resized = rescaleFrame(canny)
        cv.imshow('Video Original', frame)
        cv.imshow('Video Resized', dilateded)


        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
