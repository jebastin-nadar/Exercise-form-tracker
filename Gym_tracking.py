import cv2
import numpy as np 
from collections import deque

initial_points = []

## callback function for getting the points to track from the user
def get_intial_points(event, x, y, flags, param):
    global initial_points
    if event == cv2.EVENT_LBUTTONDOWN:
        initial_points.append([x, y])   
        
        # show a white circle on the frame to display the point which will be tracked
        cv2.circle(frame1, (x, y), 5, [255,255,255], -1) 
        cv2.imshow('Enter points to track', frame1)        


cap = cv2.VideoCapture('/path/to/video.mp4')

ret, frame1 = cap.read()
prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Get points to track from the user
cv2.imshow('Enter points to track', frame1) 
cv2.setMouseCallback('Enter points to track', get_intial_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the tracked video as output.mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = round(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame1.shape[1], frame1.shape[0]))


prev_points = np.array(initial_points).reshape(-1, 1, 2).astype(np.float32)
points = deque([initial_points], maxlen=40)
colors = [(0,0,255), (0,255,0), (0,255,255), (255,0,0)] # some random colors

# parameters for 
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


while True:
    ret, frame = cap.read()
    if frame is None:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None, **lk_params)
    
    good_new = next_points[status == 1]
    points.append(good_new.tolist())
    points_arr = np.array(points).astype(np.float32)
    
    for i in range(prev_points.shape[0]):
        for j in range(1, len(points_arr)):
            thickness = int(np.sqrt(float(j)*1.5))
            cv2.line(frame, tuple(points_arr[j - 1, :,:][i]), tuple(points_arr[j,:,:][i]),
                     colors[i], thickness)
     
    out.write(frame)
    cv2.imshow("Tracked Video", frame)
    
    # Updates previous frame
    prev_gray = gray.copy()
    prev_points = good_new.reshape(-1, 1, 2)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
