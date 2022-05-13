#Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Load images and convert them to grayscale
imgA = cv2.imread("data/imageA.png")
imgB = cv2.imread("data/imageB.png")
grayA = cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imgB,cv2.COLOR_BGR2GRAY)

#Declare SIFT object
sift = cv2.SIFT_create()

#Extracting keypoints and descriptors from the two grayscale images
kpA, descA = sift.detectAndCompute(grayA, None)
kpB, descB = sift.detectAndCompute(grayB, None)

#Declaring BFMatcher object and finding match between two images
bf = cv2.BFMatcher()
matches = bf.knnMatch(descA, descB, k=2)

#Extracting good matches
good = []
for m, n in matches:
    if m.distance < 0.7*n.distance:
        good.append([m])

#Drawing matches        
im_matches = cv2.drawMatchesKnn(imgA, kpA, imgB, kpB,
                               good[0:25], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#Extracting co-ordinates of best matches from the two images to compute Homography
src_pts  = np.float32([kpA[m[0].queryIdx].pt for m in good])
dst_pts  = np.float32([kpB[m[0].trainIdx].pt for m in good])
H, masked = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

#Function to stitch two images
def warpTwoImages(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])  # translate

    result = cv2.warpPerspective(img2, Ht@H, (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1], t[0]:w1+t[0]] = img1
    return result

#Stitching two images and storing the result
result = warpTwoImages(imgB, imgA, H)
cv2.imwrite('results/output.png', result)





