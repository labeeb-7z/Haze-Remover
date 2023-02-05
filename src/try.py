import cv2
import numpy as np


def Airlight(HazeImg, AirlightMethod, windowSize):
    if (AirlightMethod.lower() == 'fast'):
        A = []
        if (len(HazeImg.shape) == 3):
            for ch in range(len(HazeImg.shape)):
                kernel = np.ones((windowSize, windowSize), np.uint8)
                minImg = cv2.erode(HazeImg[:, :, ch], kernel)
                # cv2.imshow('Result', minImg)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                A.append(int(minImg.max()))
        else:
            kernel = np.ones((windowSize, windowSize), np.uint8)
            minImg = cv2.erode(HazeImg, kernel)
            A.append(int(minImg.max()))
    return (A)


HazeImg = cv2.imread('example4.png')
print(HazeImg.shape)

windowSze = 15
AirlightMethod = 'fast'
A = Airlight(HazeImg, AirlightMethod, windowSze)


def BoundCon(HazeImg, A, C0, C1, windowSze):
    if (len(HazeImg.shape) == 3):

        t_b = np.maximum((A[0] - HazeImg[:, :, 0].astype(np.float)) / (A[0] - C0),
                         (HazeImg[:, :, 0].astype(np.float) - A[0]) / (C1 - A[0]))
        t_g = np.maximum((A[1] - HazeImg[:, :, 1].astype(np.float)) / (A[1] - C0),
                         (HazeImg[:, :, 1].astype(np.float) - A[1]) / (C1 - A[1]))
        t_r = np.maximum((A[2] - HazeImg[:, :, 2].astype(np.float)) / (A[2] - C0),
                         (HazeImg[:, :, 2].astype(np.float) - A[2]) / (C1 - A[2]))

        MaxVal = np.maximum(t_b, t_g, t_r)
        transmission = np.minimum(MaxVal, 1)
    else:
        transmission = np.maximum((A[0] - HazeImg.astype(np.float)) / (A[0] - C0),
                                  (HazeImg.astype(np.float) - A[0]) / (C1 - A[0]))
        transmission = np.minimum(transmission, 1)

    kernel = np.ones((windowSze, windowSze), np.float)
    transmission = cv2.morphologyEx(
        transmission, cv2.MORPH_CLOSE, kernel=kernel)
    return (transmission)


windowSze = 3
C0 = 20         # Default value = 20 (as recommended in the paper)
C1 = 300        # Default value = 300 (as recommended in the paper)
# Computing the Transmission using equation (7) in the paper
Transmission = BoundCon(HazeImg, A, C0, C1, windowSze)
cv2.imshow('Result', Transmission)
cv2.waitKey(0)
cv2.destroyAllWindows()