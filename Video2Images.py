import pathlib

import cv2
import numpy as np
import os

from os.path import isfile, join


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

# https://www.life2coding.com/convert-image-frames-video-file-using-opencv-python/
#     this is the tutorial that gave and example which i used to understand converting to video better
#     Using OpenCV takes a mp4 video and produces a number of images.
#     Requirements

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
    # example
    filename=['video']

    dir='D:/imageStitchingTutorial_PyLesson/'
    #you can add different file types here such as .mp4 or ect..
    suffix=['.mov']

    for n in range(len(filename)):
        temp=filename[n]
        pathIn.append(os.path.join(dir,temp))
        try:
            if not os.path.exists(filename[n]):
                os.makedirs(filename[n])
        except OSError:
            print('Error: Creating directory of data')

    # convert_video_to_frames(filename[0], pathIn[0], suffix[0])
    # convert_video_to_frames(filename[3],pathIn[3],suffix[0])
    # convert_video_to_frames(filename[4], pathIn[4], suffix[0])
    # convert_video_to_frames(filename[5], pathIn[5], suffix[0])
    # convert_video_to_frames(filename[6], pathIn[6], suffix[0])
    # convert_video_to_frames(filename[7], pathIn[7], suffix[0])
    # convert_video_to_frames(filename[8], pathIn[8], suffix[0])
    # convert_video_to_frames(filename[9], pathIn[9], suffix[0])
    # convert_video_to_frames(filename[10], pathIn[10], suffix[0])
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




    # import fnmatch
    # for i in pathlib.Path(pathIn[0]).iterdir():
    #
    #
    #     if(fnmatch.fnmatch(i,'*.jpg')):
    #         name = str(i).split('\\')
    #         temp = name[3].split('.')
    #         test = temp[0].split('_')
    #         if (test[0] == filename[0]):
    #             images2stitch.append(name[3])
    #
    # for i in  pathlib.Path(pathIn[1]).iterdir():
    #     # files = i.split('_')
    #
    #     if (fnmatch.fnmatch(i, '*.jpg')):
    #         name=str(i).split('\\')
    #         temp=name[3].split('.')
    #         test=temp[0].split('_')
    #         if(test[0]==filename[1]):
    #           images2stitchalso.append(name[3])
    #
    # temp=0

    fps=25
    convert_frames_to_video(pathIn[0],pathOut,fps)


if __name__=="__main__":
    main()