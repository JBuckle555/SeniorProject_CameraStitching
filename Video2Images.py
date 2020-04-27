import pathlib

import cv2
import numpy as np
import os

from os.path import isfile, join
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


def Stitching(image1Name,image2Name,WindowName):
    try:
        if not os.path.exists(WindowName):
            os.makedirs(WindowName)
    except OSError:
        print('Error: Creating directory of data')
    temp=["matches","trim"]
    test=image1Name.split('_')
    test2=image2Name.split('_')
    img_ = cv2.imread(test[0]+'/'+image1Name)

    img_ = cv2.resize(img_, (560, 240), fx=1, fy=1)
    img1 = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)

    img = cv2.imread(test2[0]+'/'+image2Name)
    img = cv2.resize(img, (560, 240), fx=1, fy=1)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    # find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    #cv2.imshow('original_image1_keypoints', cv2.drawKeypoints(img_, kp1, None))
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
    #cv2.namedWindow(WindowName + temp[0])
    #img3 = cv2.drawMatches(img_, kp1, img, kp2, good, None, **draw_params)
    #cv2.imshow(WindowName+temp[0], img3)
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

    # cv2.namedWindow(WindowName+temp[1])
    # cv2.imshow(WindowName+temp[1], trim(dst))
    # cv2.imshow(WindowName+temp[1],trim(dst))
    # cv2.waitKey(30)
    cv2.imwrite(WindowName+temp[1]+'.jpg',trim(dst),None)
    # cv2.waitKey(2000)
    trim(dst)

def convert_video_to_frames(filename,pathout,suffix):
    '''
    https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481
    is the tutorial i used to understand how to make video to images
    
    ----
    You require OpenCV 3.2 to be installed.
    Run
    ----
    Open the main.py and edit the path to the video. Then run:
    $ python main.py
    Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
    '''

    # Playing video from file:
    cap = cv2.VideoCapture(filename+suffix)

    currentFrame = 0
    print('%s start' %filename)
    while (cap.isOpened()):
        # cap.set(cv2.CAP_PROP_POS_MSEC,(currentFrame*1000))
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret==False:
            break

        # Saves image of the current frame in jpg file
        name = filename+ '_'+str(currentFrame) + '.jpg'

        # print('Creating: %s' % name)
        cv2.imwrite(os.path.join(pathout , name), frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    print('%s end' % filename)
    cap.release()
    cv2.destroyAllWindows()

def convert_frames_to_video(pathIn, pathOut, fps):
'''
https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
    this is the tutorial that gave and example which i used to understand converting to video better
    Using OpenCV takes a mp4 video and produces a number of images.
    Requirements
'''
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))
    temp=files.split('_')


    for i in range(len(files)):
        if (temp[0] != 'video'):
            files.remove(i)
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()



def main():
    pathIn = []
    pathOut = 'video.avi'
    filename=['9test1','9test2','Video',
              '1test1','2test1','3test1','4test1','5test1','6test1','7test1','8test1',
              '1test2','2test2','3test2','4test2','5test2','6test2','7test2','8test2','10test2']
    dir='D:/imageStitchingTutorial_PyLesson/'
    suffix='.mov'

    for n in range(len(filename)):
        temp=filename[n]
        pathIn.append(os.path.join(dir,temp))
        try:
            if not os.path.exists(filename[n]):
                os.makedirs(filename[n])
        except OSError:
            print('Error: Creating directory of data')

    # convert_video_to_frames(filename[0], pathIn[0], suffix)
    # convert_video_to_frames(filename[3],pathIn[3],suffix)
    # convert_video_to_frames(filename[4], pathIn[4], suffix)
    # convert_video_to_frames(filename[5], pathIn[5], suffix)
    # convert_video_to_frames(filename[6], pathIn[6], suffix)
    # convert_video_to_frames(filename[7], pathIn[7], suffix)
    # convert_video_to_frames(filename[8], pathIn[8], suffix)
    # convert_video_to_frames(filename[9], pathIn[9], suffix)
    # convert_video_to_frames(filename[10], pathIn[10], suffix)
    #
    #
    # convert_video_to_frames(filename[1],pathIn[1],suffix)
    # convert_video_to_frames(filename[11], pathIn[11], suffix)
    # convert_video_to_frames(filename[12], pathIn[12], suffix)
    # convert_video_to_frames(filename[13], pathIn[13], suffix)
    # convert_video_to_frames(filename[14], pathIn[14], suffix)
    # convert_video_to_frames(filename[15], pathIn[15], suffix)
    # convert_video_to_frames(filename[16], pathIn[16], suffix)
    # convert_video_to_frames(filename[17], pathIn[17], suffix)
    # convert_video_to_frames(filename[18], pathIn[18], suffix)
    # convert_video_to_frames(filename[19], pathIn[19], suffix)
    #imagenames['test1.jpg','example.jpg']
    #Stitching(imagenames[0],imagenames[1],'windowname',)

    images2stitch=[]
    images2stitchalso=[]
    count=0


    import fnmatch
    for i in pathlib.Path(pathIn[0]).iterdir():


        if(fnmatch.fnmatch(i,'*.jpg')):
            name = str(i).split('\\')
            temp = name[3].split('.')
            test = temp[0].split('_')
            if (test[0] == filename[0]):
                images2stitch.append(name[3])

    for i in  pathlib.Path(pathIn[1]).iterdir():
        # files = i.split('_')

        if (fnmatch.fnmatch(i, '*.jpg')):
            name=str(i).split('\\')
            temp=name[3].split('.')
            test=temp[0].split('_')
            if(test[0]==filename[1]):
              images2stitchalso.append(name[3])

    temp=0
    if (len(images2stitchalso)<len(images2stitch)):
        count=len(images2stitchalso)
    else:
        count=len(images2stitch)
    while(count>0):

        test = images2stitch.pop()
        test2= images2stitchalso.pop()
        half=test.split('.')
        half2=test2.split('.')
        step=half[0].split('_')
        step2=half2[0].split('_')
        imagename=half[0]
        imagename2=half2[0]

        if(step[1]==step2[1] ):
            # print(images2stitch[temp])
            # print(images2stitchalso[temp])

            Stitching(pathIn[0]+imagename[temp],pathIn[1]+imagename2[temp],filename[2],pathIn[2])
            temp+=1
            count-=1
        else:
            images2stitch.append(test)
            images2stitchalso.append(test2)
    fps=25
    # convert_frames_to_video(pathIn[2],pathOut,fps)


if __name__=="__main__":
    main()