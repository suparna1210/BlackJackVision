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
import cv2
import numpy as np
import time
import os
import funcs
import video

# frame constants
IM_WIDTH = 1280
IM_HEIGHT = 720

# creating a video streaming object using the video class
stream = video.video((IM_WIDTH,IM_HEIGHT),10,0).start()
# break before opening camera
time.sleep(1)

# card rank and suits - train datasets
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = funcs.ranks( path + '/Card_Imgs/')
train_suits = funcs.suits( path + '/Card_Imgs/')

# loop control variable
flag = 0

#variables to return to function - usage in Blackjack module
no_of_cards = 0
cards = []

# frame capture loop
while flag == 0:

    # frame-by-frame from video stream
    frame = stream.read()

    # pre-processing function - grayscale, blur, threshold
    preprocessed = funcs.preprocess(frame)
	
    # finding card contours
    sorted_contours, is_card_contour = funcs.find_cards(preprocessed)

    # if number of cards is not 0
    if len(sorted_contours) != 0:

        # cards - card object list
        # k - number of cards (index)
        cards = []
        k = 0

        # for each contour:
        for i in range(len(sorted_contours)):
            if (is_card_contour[i] == 1):

                cards.append(funcs.preprocess_card(sorted_contours[i],frame))

                # matching the best rank and suit of the current card
                cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = funcs.match(cards[k],train_ranks,train_suits)

                # drawing rank and suit result on cards
                frame = funcs.draw(frame, cards[k])
                k = k + 1
	    
        # drawing bounding boxes
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                # temporary contour list to draw
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(frame,temp_cnts, -1, (255,0,0), 2)
        no_of_cards = k
        cards = cards   

    # display identified cards
    cv2.imshow("Card Detector",frame)

    # exit main loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        flag = 1

# once main loop is exited, store card results (rank only)        
if(flag == 1):
    funcs.store_result(no_of_cards, cards)

# Close all windows and video stream
cv2.destroyAllWindows()
stream.stop()

