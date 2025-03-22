import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# color for mask 
lower_blue = np.array([100, 50, 50], dtype="uint8")
upper_blue = np.array([140, 255, 255], dtype="uint8")

def adjust_brightness_contrast(image, alpha=1.0, beta=30):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def calculate_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])  
    return brightness

def adjust_brightness(image, target_brightness=150):
    
    current_brightness = calculate_brightness(image)
    brightness_difference = target_brightness - current_brightness

    
    if brightness_difference > 20:  # dark
        image = adjust_brightness_contrast(image, alpha=1.0, beta=int(brightness_difference))
        return image, #Increasing Brightness
    elif brightness_difference < -20:  # bright
        image = adjust_brightness_contrast(image, alpha=1.0, beta=int(brightness_difference))
        return image, #Decreasing Brightness
    return image, 

def smooth_image(image):
    #smoothing image
    return cv2.GaussianBlur(image, (5, 5), 0)

def edge_detection(image):
    return cv2.Canny(image, 100, 200)

def align_face(frame, face_rect):
   #aligning
    (x, y, w, h) = face_rect
    face_roi = frame[y:y + h, x:x + w]

    # image coverted to  grayscale 
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(eyes) >= 2: 
        eyes = sorted(eyes, key=lambda ex: ex[0])
        left_eye = eyes[0]
        right_eye = eyes[1]
        
        left_eye_center = (int(left_eye[0] + left_eye[2] / 2), int(left_eye[1] + left_eye[3] / 2))
        right_eye_center = (int(right_eye[0] + right_eye[2] / 2), int(right_eye[1] + right_eye[3] / 2))

       # detect angle between eyes
        delta_x = right_eye_center[0] - left_eye_center[0]
        delta_y = right_eye_center[1] - left_eye_center[1]
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        # detecting center between the two eyes
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
     #measuring distance between eyes
     
        eye_distance = np.sqrt(delta_x ** 2 + delta_y ** 2)
        desired_distance = 150  
        scale = desired_distance / eye_distance
        
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=scale)

        # Rotating the entire face
        aligned_face = cv2.warpAffine(face_roi, rotation_matrix, (w, h))

        cv2.putText(frame, f"Angle: {angle:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 
        return aligned_face, angle
    else:
       # if eye not detected
        return face_roi, 0  

def detect_face_mask(frame):
    # adujusting the brightness according to light
    frame, brightness_msg = adjust_brightness(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # to detect number of faces in a frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    rotation_angle = 0  

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Alignment of face 
        aligned_face, angle = align_face(frame, (x, y, w, h))
        rotation_angle = angle  

        #check for lower face detection
        lower_half = aligned_face[h // 2:h, 0:w]

        # change color for better detection
        hsv = cv2.cvtColor(lower_half, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_area = cv2.countNonZero(mask)
        total_area = lower_half.shape[0] * lower_half.shape[1]
        mask_percentage = (mask_area / total_area) * 100

        if mask_percentage > 10:  
            cv2.putText(frame, "Mask Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Mask", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # message for brightness adjustment
    cv2.putText(frame, brightness_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame, rotation_angle 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect face and mask
    output_frame, rotation_angle = detect_face_mask(frame)

    cv2.putText(output_frame, f"Rotation Angle: {rotation_angle:.2f}", 
                (frame.shape[1] - 250, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 

    cv2.imshow("Face Mask Detection", output_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()