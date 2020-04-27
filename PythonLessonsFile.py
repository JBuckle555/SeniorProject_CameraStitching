import cv2
import numpy as np
# the stitching and trim came from Pylessons tutorial on image stitching which i modified to 
#allow me to control what images were bing made
def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    elif not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    elif not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    elif not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame


def Stitching(image1Name,image2Name,WindowName,PicNames):
    temp=["matches","trim"]

    img_ = cv2.imread(image1Name)

    img_ = cv2.resize(img_, (560, 240), fx=1, fy=1)
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(image2Name)
    img = cv2.resize(img, (560, 240), fx=1, fy=1)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    cv2.imshow('original_image1_keypoints', cv2.drawKeypoints(img_, kp1, None))
    #cv2.imshow('original_image2_keypoints', cv2.drawKeypoints(img, kp2, None))

    match = cv2.BFMatcher()
    matches = match.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 1 * n.distance:
            good.append(m)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)
    cv2.namedWindow(WindowName + temp[0])
    img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)
    cv2.imshow(WindowName+temp[0], img3)
    # cv2.waitKey(0)

    MIN_MATCH_COUNT = 1

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w = img2.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        #img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        #cv2.imshow("original_image_overlapping.jpg", img2)
        # cv2.waitKey(0)

    else:
        print("Not enough matches are found - %d/%d", (len(good) / MIN_MATCH_COUNT))

    dst = cv2.warpPerspective(img_, M, (img_.shape[1] + img_.shape[1], img.shape[0]))
    dst[0:img.shape[0], 0:img.shape[1]] = img
    # cv2.imshow("original_image_stitched.jpg", dst)
    #cv2.waitKey(0)

    cv2.namedWindow(WindowName+temp[1])
    # trim(dst)
    # cv2.imshow(WindowName+temp[1], trim(dst))
    cv2.imshow(WindowName+temp[1],trim(dst))
    cv2.waitKey(30)
    cv2.imwrite(PicNames+WindowName+temp[1]+'.jpg',trim(dst),None)
    cv2.waitKey(2000)

    # cv2.imwrite("original_image_stiched_crop.jpg", trim(dst))
#imageNames=["original_image_left.jpg","original_image_right2.jpg"]
# imageNames=["ACM1.jpg","ACM2.jpg"]
#Stitching(imageNames[0],imageNames[1])
#imageNames=["Book.jpg","Book2.jpg","Book3.jpg"]
#imageNames=["Face1.jpg","Face2.jpg"]
# imageNames=['board1.jpg','board2.jpg']
imageNames=['5test1_291.png','7test2_291.png']
WindowNames=["1","2","3"]
PicNames=['ACM','Book','Face','Board']





Stitching(imageNames[0],imageNames[1],WindowNames[0],PicNames[3])
Stitching(imageNames[1],imageNames[0],WindowNames[1],PicNames[3])
# Stitching(imageNames[0],imageNames[1],WindowNames[0],PicNames[0])
# Stitching(imageNames[1],imageNames[0],WindowNames[0],PicNames[0])
# Stitching(imageNames[1],imageNames[0],WindowNames[1])

#imageNames=["Book.jpg","Book2.jpg","Book3.jpg"]
#Stitching(imageNames[0],imageNames[2],WindowNames[0],PicNames[1])
#Stitching(imageNames[1],imageNames[2],WindowNames[1],PicNames[1])
#Stitching(imageNames[1],imageNames[0],WindowNames[2],PicNames[1])

'''
for i in range(len(imageNames)):
    if(i<2):
        Stitching(imageNames[i],imageNames[i+1],WindowNames[i])
    else:
        Stitching(imageNames[i],imageNames[0],WindowNames[i])
        
'''