import cv2

# Pre-trained classifiers 
classifierC = 'car_detector.xml'
classifierP = 'pedest_detector.xml'

# Get video footage
vid = cv2.VideoCapture('cars&p.mov')

# Create car and pedestrian classifier
car_tracker = cv2.CascadeClassifier(classifierC)
p_tracker = cv2.CascadeClassifier(classifierP)

# Iterate forever over frames
while True:
   
    # Read current frame
    (read_successful, frame) = vid.read()

    # Safe coding
    if read_successful:
        # Convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break 

    # Detect Cars and Pedestrians, detectMultiScale detects diff sizes and returns coordinates of rectangles surrounding face
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = p_tracker.detectMultiScale(grayscaled_frame) 

    # Draw rectangles around the cars and pedestrians, (0,255,0) is RGB colour, 2 is thickness of rectangle
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Show video w rectangles
    cv2.imshow('Car & Pedestrian Detector', frame)

    # Waits for key press to close
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy any open windows
vid.release()
cv2.destroyAllWindows()
