import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize variables for color points
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# Indexes to mark points in arrays of specific colors
blue_index = green_index = red_index = yellow_index = 0


kernel = np.ones((5, 5), np.uint8)

# Colors for drawing
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
colorIndex = 0


paintWindow = np.zeros((471, 636, 3)) + 255
regions = [(40, 140), (160, 255), (275, 370), (390, 485), (505, 600)]
labels = ["CLEAR", "BLUE", "GREEN", "RED", "YELLOW"]

for (start, end, label) in zip(regions, regions[1:], labels):
    paintWindow = cv2.rectangle(paintWindow, start, end, (0, 0, 0), 2)
    cv2.putText(paintWindow, label, (start[0] + 9, start[1] + 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
ret = True

while ret:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40, 1), (600, 65), (0, 0, 0), 2)
    cv2.putText(frame, "BRUSH SIZE: {}".format(len(bpoints[blue_index])), (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    result = hands.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = [(
            int(lm.x * 640),
            int(lm.y * 480)
        ) for handslms in result.multi_hand_landmarks for lm in handslms.landmark]

        mpDraw.draw_landmarks(frame, result.multi_hand_landmarks[0], mpHands.HAND_CONNECTIONS)

        fore_finger = landmarks[8]
        center = fore_finger
        thumb = landmarks[4]

        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        if thumb[1] - center[1] < 30:
            bpoints.append(deque(maxlen=512))
            gpoints.append(deque(maxlen=512))
            rpoints.append(deque(maxlen=512))
            ypoints.append(deque(maxlen=512))

            blue_index += 1
            green_index += 1
            red_index += 1
            yellow_index += 1

        elif center[1] <= 65:
            for i, (start, end) in enumerate(zip(regions, regions[1:])):
                if start[0] <= center[0] <= end[0]:
                    if i == 0:  
                        bpoints = [deque(maxlen=512)]
                        gpoints = [deque(maxlen=512)]
                        rpoints = [deque(maxlen=512)]
                        ypoints = [deque(maxlen=512)]

                        blue_index = green_index = red_index = yellow_index = 0
                        paintWindow[67:, :] = 255
                    else:
                        colorIndex = i - 1

        elif brush_size > 1:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)

    else:
        bpoints.append(deque(maxlen=512))
        gpoints.append(deque(maxlen=512))
        rpoints.append(deque(maxlen=512))
        ypoints.append(deque(maxlen=512))

        blue_index += 1
        green_index += 1
        red_index += 1
        yellow_index += 1

    # Draw lines on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]

    for i, color_points in enumerate(points):
        for j in range(len(color_points[blue_index])):
            for k in range(1, len(color_points[blue_index][j])):
                if color_points[blue_index][j][k - 1] is not None and color_points[blue_index][j][k] is not None:
                    cv2.line(frame, color_points[blue_index][j][k - 1], color_points[blue_index][j][k], colors[i], 2)
                    cv2.line(paintWindow, color_points[blue_index][j][k - 1], color_points[blue_index][j][k], colors[i], 2)

    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("drawing.png", paintWindow)
        print("Drawing saved as 'drawing.png'")
    elif key == ord('l'):
        loaded_drawing = cv2.imread("drawing.png")
        paintWindow[67:, :] = 255 
        paintWindow[67:, :] = loaded_drawing

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
