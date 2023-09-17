#############################
#
# CS 5330 - Pattern Recognition and Computer Vision
# Final Project - Blackjack game
# Team members: Aparna Krishnan, Farhan Sarfraz, Suparna Srinivasan
# Spring 2022
# Source: https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
#
#############################

# import statements
from threading import Thread
import cv2


class video:
    def __init__(self, resolution=(640,480),framerate=30,src=0):
      
        self.stream = cv2.VideoCapture(src)
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
        (self.grabbed, self.frame) = self.stream.read()

	# variable for when camera is stopped
        self.stopped = False

    def start(self):
	# thread to start reading video frames
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
            while True:
                if self.stopped:
                    self.stream.release()
                    return
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
