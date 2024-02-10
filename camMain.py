import cv2
import numpy as np
from functions import *
import sudukoSolver

# Suppress TensorFlow warning messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    print('Setting UP')
    
    # Load the CNN Model
    model = intializePredectionModel()

    # Initialize camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display the camera feed
        cv2.imshow('Camera', frame)

        # Press 's' to capture the Sudoku puzzle
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Changing Height and Width to make it square
            heightImg = 450
            widthImg = 450

            # Preprocess the input image
            img = cv2.resize(frame, (widthImg, heightImg))
            imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
            imgThreshold = preProcess(img)

            # Find contours
            imgContours = img.copy()
            imgBigContour = img.copy()
            contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            biggest, maxArea = biggestContour(contours)

            if biggest.size != 0:
                # Preprocess the Sudoku grid
                biggest = reorder(biggest)
                pts1 = np.float32(biggest)
                pts2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
                imgDetectedDigits = imgBlank.copy()
                imgWarpColored = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)

                # Split the Sudoku grid into individual cells
                boxes = splitBoxes(imgWarpColored)
                numbers = getPredection(boxes, model)

                # Display detected numbers
                imgDetectedDigits = displayNumbers(imgDetectedDigits, numbers)
                numbers = np.asarray(numbers)
                posArray = np.where(numbers > 0, 0, 1)

                # Solve the Sudoku puzzle
                board = np.array_split(numbers, 9)
                sudukoSolver.solve_sudoku(board)

                flatList = []
                for sublist in board:
                    for item in sublist:
                        flatList.append(item)
                solvedNumbers = flatList * posArray
                imgSolvedDigits = displayNumbers(imgBlank.copy(), solvedNumbers)

                # Overlay the solution on the original image
                pts2 = np.float32(biggest)
                pts1 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgInvWarpColored = cv2.warpPerspective(imgSolvedDigits, matrix, (widthImg, heightImg))
                inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, img, 0.5, 1)
                imgDetectedDigits = drawGrid(imgDetectedDigits)
                imgSolvedDigits = drawGrid(imgSolvedDigits)

                # Display the processed frame
                imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                              [imgDetectedDigits, imgSolvedDigits, imgInvWarpColored, inv_perspective])
                stackedImage = stackImages(imageArray, 1)
                cv2.imshow('Sudoku Solver', stackedImage)
                cv2.waitKey(0)  # Wait indefinitely until a key is pressed

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
