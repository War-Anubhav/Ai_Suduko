War-Anubhav/Ai_Suduko# Sudoku Solver

## Overview
This project implements a Sudoku Solver using computer vision techniques and deep learning. It utilizes image processing to extract the Sudoku puzzle from an image and then employs a convolutional neural network (CNN) to recognize and solve the puzzle. The solution is then overlaid onto the original image, providing a visual representation of the solved puzzle.

## Features
- Extracts Sudoku puzzle grids from images
- Recognizes digits within the puzzle using a pre-trained CNN model
- Solves the Sudoku puzzle
- Displays the solved puzzle overlaid on the original image

## Prerequisites
- Python 3.x
- OpenCV (cv2)
- NumPy
- TensorFlow (for loading and using the pre-trained CNN model)

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/sudoku-solver.git
   ```
2. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Ensure that the Sudoku puzzle images are stored in the `Resources` directory.
2. Run the `Main.py` script:
   ```
   python Main.py
   ```
3. The script will process the Sudoku puzzle images, recognize digits, solve the puzzles, and display the results.

## Directory Structure
- `functions.py`: Contains helper functions for image processing, digit recognition, and Sudoku solving.
- `Main.py`: Main script for running the Sudoku Solver.
- `Resources/`: Directory for storing Sudoku puzzle images.

## Credits
This project was developed by Anubhav Ranjan.
