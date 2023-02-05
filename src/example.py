import cv2
import numpy as np

from alright import Airlight
from boundcon import BoundCon
from caltransmission import CalTransmission
from removehaze import removeHaze

if __name__ == '__main__':
    HazeImg = cv2.imread('../assets/example1.png')

    # Resize image
    
    # Channels = cv2.split(HazeImg)
    # rows, cols = Channels[0].shape
    # HazeImg = cv2.resize(HazeImg, (int(2.4 * cols), int(2.4 * rows)))
    

    # Estimate Airlight
    windowSze = 15
    AirlightMethod = 'fast'
    A = Airlight(HazeImg, AirlightMethod, windowSze)

    # Calculate Boundary Constraints
    windowSze = 3
    C0 = 20         # ecommended in the paper
    C1 = 300        # recommended in the paper
    Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)

    # Refine estimate of transmission
    # Default value = 1 (as recommended in the paper) --> Regularization parameter, the more this  value, the closer to the original patch wise transmission
    regularize_lambda = 1
    sigma = 0.5
    # Using contextual information
    Transmission = CalTransmission(
        HazeImg, Transmission, regularize_lambda, sigma)

    # Perform DeHazing
    HazeCorrectedImg = removeHaze(HazeImg, Transmission, A, 0.85)

    cv2.imshow('Original', HazeImg)
    cv2.imshow('Result', HazeCorrectedImg)
    cv2.waitKey(0)

    #cv2.imwrite('outputImages/result.jpg', HazeCorrectedImg)
