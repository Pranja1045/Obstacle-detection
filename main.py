from time import sleep
#import gpiozero's Robot since we no longer use motors
import cv2
import numpy as np
import math
import os
# Initialize constants
StepSize = 5
currentFrame = 0
testmode = 1  # 1: Save and log data | 2: Show frames only

# Create output directory
if not os.path.exists('data'):
    os.makedirs('data')

if testmode == 1:
    F = open("./data/imagedetails.txt", 'a')
    F.write("\n\nNew Test \n")

def calc_dist(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def getChunks(l, n):
    a = []
    for i in range(0, len(l), n):
        a.append(l[i:i + n])
    return a

# Start camera
cap = cv2.VideoCapture(1)

while True:
    _, frame = cap.read()
    name = './data/frame' + str(currentFrame) + '.jpg'
    name1='./dataset/train/frame' + str(currentFrame) + '.jpg'
    name2='./dataset/val/frame' + str(currentFrame) + '.jpg'
    labeler=open('./dataset/label/frame' + str(currentFrame) + '.txt','w')
    print('Processing...', name)

    img = frame.copy()
    # thanks Pranshu Raj for help

    # Smooth and detect edges
    blur = cv2.bilateralFilter(img, 9, 40, 40)
    edges = cv2.Canny(blur, 50, 100)

    img_h = img.shape[0] - 1
    img_w = img.shape[1] - 1

    EdgeArray = []

    # Extract edge points from bottom-up
    for j in range(0, img_w, StepSize):
        pixel = (j, 0)
        for i in range(img_h - 5, 0, -1):
            if edges.item(i, j) == 255:
                pixel = (j, i)
                break
        EdgeArray.append(pixel)

    # Draw lines between edge points
    for x in range(len(EdgeArray) - 1):
        cv2.line(img, EdgeArray[x], EdgeArray[x + 1], (0, 255, 0), 1)

    for x in range(len(EdgeArray)):
        cv2.line(img, (x * StepSize, img_h), EdgeArray[x], (0, 255, 0), 1)

    # Divide into chunks and compute average edge
    chunks = getChunks(EdgeArray, int(len(EdgeArray) / 3))
    c = []

    for i in range(len(chunks) - 1):
        x_vals = [x for (x, _) in chunks[i]]
        y_vals = [y for (_, y) in chunks[i]]
        avg_x = int(np.average(x_vals))
        avg_y = int(np.average(y_vals))
        c.append([avg_y, avg_x])
        cv2.line(frame, (320, 480), (avg_x, avg_y), (255, 0, 0), 2)

    if len(c) < 3:
        print("Not enough segments detected.")
        continue

    forwardEdge = c[1]
    y = min(c)

    # Mark forward edge
    cv2.line(frame, (320, 480), (forwardEdge[1], forwardEdge[0]), (0, 255, 0), 3)
    obstacle_detected=False
    # Determine direction (no motor movement)
    if forwardEdge[0] > 250:
        obstacle_detected=True
        if y[1] < 310:
            direction = "Obstacle: LEFT"
        else:
            direction = "Obstacle: RIGHT"
    else:
        direction = "Path: FORWARD"

    print(direction)
    if obstacle_detected:
        box_size = 200
        top_left = (forwardEdge[1] - box_size // 2, forwardEdge[0] - box_size // 2)
        bottom_right = (forwardEdge[1] + box_size // 2, forwardEdge[0] + box_size // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(frame, "Detected", (top_left[0], top_left[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        x_min,y_min=top_left
        x_max,y_max=bottom_right
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2
        width = x_max - x_min
        height = y_max - y_min
        x_center/=img_w
        y_center/=img_h
        
        labeler.write(f"0 {x_center:.6f} {y_center:6f} {width:6f} {height:6f}\n")
        
    else:
        pass
    # Color-coded direction suggestion
    if "FORWARD" in direction:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    cv2.putText(frame, direction, (150, 470),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Save frame and log
    cv2.imwrite(name, frame)
    
        
    if testmode == 1:
        F.write("frame" + str(currentFrame) + ".jpg | " +
                str(c[0]) + " | " + str(c[1]) + " | " +
                str(c[2]) + " | " + direction + "\n")
         
        currentFrame += 1
        resized=cv2.resize(frame,(500,500))
        if currentFrame <1000:
            cv2.imwrite(name1, frame)
        else:
            cv2.imwrite(name2, frame)

        cv2.imshow("Original Frame", resized)

    if testmode == 2:
        resized=cv2.resize(frame,(500,500))
        resized1=cv2.resize(edges,(500,500))
        cv2.imshow("Original Frame", resized)
        cv2.imshow("Canny Edges", resized1)

    if cv2.waitKey(5) & 0xFF == 27:  # Press Esc to quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if testmode == 1:
    F.close()
    labeler.close()
