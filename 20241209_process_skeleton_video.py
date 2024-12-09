#landmarks_tuple = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
#    reba_score = reba.getRebaScore(landmarks_tuple)

#    cv.putText(image, f'Upper right arm angle: {int(upper_arm)} ,{reba_score}', (50, 50), 
#                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#    cv.imshow('Ergonomics Assessment', image)






#Import libraries
import os
#os.chdir("C:/Users/raszt/OneDrive/Asztali gép/Szakdolgozat/REBA_2D_calculation")
os.chdir("C:/Users/Tuan-anh/OneDrive - Pannon Egyetem/RMIT/Projects/Flex-ergonomy-2024/REBA_2D_calculation")
import numpy as np
import cv2 as cv
import mediapipe as mp
import math
from matplotlib import pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime, timedelta
# import calculate_REBA as reba
# from ergonomics.reba import RebaScore
from RULA_calculator import GetRULAScores as rc


def calculate_MiddlePoint(p1 , p2):

    p1 = np.array(p1)
    p2 = np.array(p2)

    midpoint = (p1 + p2) / 2

    return tuple(midpoint)

# Define a function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Ensure angle is between 0 and 180
    # if angle > 180.0:
    #     angle = 360 - angle

    return angle


def get_landmark_coords(landmark):
    return (landmark.x, landmark.y, landmark.z)

def calculate_angle_between_vectors(a, b, c, d):
    """
    Calculate the angle between vectors AB and CD.
    
    Parameters:
    a (tuple): Coordinates of point A (x1, y1).
    b (tuple): Coordinates of point B (x2, y2).
    c (tuple): Coordinates of point C (x3, y3).
    d (tuple): Coordinates of point D (x4, y4).
    
    Returns:
    float: The angle in degrees.
    """
    # Vector AB
    ab = (b[0] - a[0], b[1] - a[1])
    # Vector CD
    cd = (d[0] - c[0], d[1] - c[1])
    
    # Dot product of AB and CD
    dot_product = ab[0] * cd[0] + ab[1] * cd[1]
    # Magnitudes of AB and CD
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_cd = math.sqrt(cd[0]**2 + cd[1]**2)
    
    # Cosine of the angle
    cos_angle = dot_product / (magnitude_ab * magnitude_cd)
    
    # Avoid floating-point errors outside the range [-1, 1]
    cos_angle = max(-1, min(1, cos_angle))
    
    # Calculate the angle in radians and convert to degrees
    angle = math.acos(cos_angle)
    return math.degrees(angle)
    
#%################################################################################ 
# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(    min_detection_confidence=0.5,
                        min_tracking_confidence=0.5     ) 
mp_drawing_styles = mp.solutions.drawing_styles
total_frames = 0
#%%###############################################################################
# Processing one by one
video_name = 'IMG_1017.MOV'

#% Start video streaming
cam = cv.VideoCapture(video_name)
# Video setting
width= int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
height= int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
# Reduce image size (downscale for speed)
scale_factor = 0.5  # Downscale factor (0.5 means reducing the size by half)
downscaled_width = int(width * scale_factor)
downscaled_height = int(height * scale_factor)

frame_count = 0  # Frame counter for time tracking
fps=30
# create visualization
blank_canvas = np.zeros((downscaled_height, int(downscaled_width*2), 3), dtype=np.uint8)
vid_writer = cv.VideoWriter(video_name[:-4]+"_output.avi",cv.VideoWriter_fourcc(*'XVID'), fps, (blank_canvas.shape[1], downscaled_height))
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

while True:
    _ret, frame = cam.read()
    if not _ret:
        print('No frames grabbed!')
        break
    
    # Resize frame for faster processing
    frame = cv.resize(frame, (downscaled_width, downscaled_height))
    # Convert the BGR image to RGB.
    image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    # process the RGB frame to get the skeleton
    results = pose.process(image)
    #print(results.pose_landmarks)
    # Convert back to BGR for rendering
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    
    # Check if pose landmarks are detected.
    if results.pose_landmarks:
        # Extract relevant landmarks
        landmarks = results.pose_landmarks.landmark

        # Shoulder, Hip, and Ear coordinates (assuming back view)
        nose = [landmarks[mp_pose.PoseLandmark.NOSE].x ,landmarks[mp_pose.PoseLandmark.NOSE].y , landmarks[mp_pose.PoseLandmark.NOSE].z]
        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP].z]
        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z]
        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y , landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z]
        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z]
        left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR].y]
        right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y , landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].z]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y , landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].z]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x , 
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y , landmarks[mp_pose.PoseLandmark.LEFT_WRIST].z]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x , 
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y ,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].z]
        left_ankle =  [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x , landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y , landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].z]
        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x ,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y , landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].z]
        left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x , 
                       landmarks[mp_pose.PoseLandmark.LEFT_PINKY].y]
        left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x , 
                       landmarks[mp_pose.PoseLandmark.LEFT_INDEX].y]
        right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x , 
                       landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].y]
        right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x , 
                       landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].y]
        
        
        

        ear_center = calculate_MiddlePoint(left_ear , right_ear)      
        ankle_center = calculate_MiddlePoint(left_ankle , right_ankle)
        #print('Head:' , head)  
                   
        # Calculate mid-points
        ear_center = calculate_MiddlePoint(left_ear , right_ear)
        shoulder_center = calculate_MiddlePoint(left_shoulder , right_shoulder)
        hip_center = calculate_MiddlePoint(left_hip, right_hip)
        knee_center = calculate_MiddlePoint(left_knee , right_knee)
        hand_center_l = calculate_MiddlePoint(left_pinky , left_index)
        hand_center_r = calculate_MiddlePoint(right_pinky , right_index)
        
        # angles
        # upper_arm_r = calculate_angle(right_elbow, right_shoulder, right_hip)
        # upper_arm_l = calculate_angle(left_elbow, left_shoulder, left_hip)
        
        upper_arm_r = calculate_angle_between_vectors(right_shoulder, right_elbow , shoulder_center, hip_center)
        upper_arm_l = calculate_angle_between_vectors(left_shoulder, left_elbow , shoulder_center, hip_center)
        
        lower_arm_r = calculate_angle_between_vectors(right_elbow , right_wrist, right_elbow, right_shoulder)
        lower_arm_l = calculate_angle_between_vectors(left_elbow , left_wrist, left_elbow, left_shoulder)
        
        wrist_l = abs(180 - calculate_angle(left_elbow , left_wrist, hand_center_l))
        wrist_r = abs(180 - calculate_angle(right_elbow , right_wrist, hand_center_r))

        neck_angle = abs(180 - calculate_angle(hip_center, shoulder_center , ear_center))
        trunk_angle = abs(180 - calculate_angle(knee_center, hip_center, shoulder_center))

        """
        #creating the structure for REBA
        #landmarks_tuple = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
        landmark_list = []
        landmark_list.append(nose)
        landmark_list.append(nose)
        landmark_list.append(left_shoulder)
        landmark_list.append(left_elbow)
        landmark_list.append(left_wrist)
        landmark_list.append(right_shoulder)
        landmark_list.append(right_elbow)
        landmark_list.append(right_wrist)
        landmark_list.append(left_hip)
        landmark_list.append(left_knee)
        landmark_list.append(left_ankle)
        landmark_list.append(right_hip)
        landmark_list.append(right_knee)
        landmark_list.append(right_ankle)
        print('List: ', landmark_list)
        #print('Nose: ',nose)
        #print('left hip:' , left_hip)
        # Draw landmarks and connections
        rebascore = RebaScore()
        """
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        image = cv.line(image, (int(ear_center[0]),int(ear_center[1])), (int(left_shoulder[0]),int(right_shoulder[1])), (255, 0, 0), 2)
    
    # Display the image
    # cv.putText(image, f'Upper right arm angle: {int(upper_arm)}, RULA score {int(upper_arm_score)}', (50, 50), 
     #           cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    # landmark = landmarks[:-1]
    
    # create visualization
    blank_canvas = np.zeros((downscaled_height, int(downscaled_width*2), 3), dtype=np.uint8)

    blank_canvas[0 : downscaled_height, 0 : downscaled_width] = image
    cv.putText(image, f'Upper right arm angle: {int(upper_arm_l)}', (50, 50), 
                 cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv.putText(blank_canvas, 'A. UPPER ARM ANGLE: ', (downscaled_width + 50, 100),
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2 )
    
    cv.putText(blank_canvas, f'Right arm: {int(upper_arm_r)}', (downscaled_width + 50, 150), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv.putText(blank_canvas, f'Left arm: {int(upper_arm_l)}', (downscaled_width + 50, 200), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    cv.putText(blank_canvas, 'A. LOWER ARM ANGLE: ', (downscaled_width + 50, 250), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, f'Right arm: {int(lower_arm_r)}', (downscaled_width + 50, 300), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv.putText(blank_canvas, f'Left arm: {int(lower_arm_l)}', (downscaled_width + 50, 350), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    cv.putText(blank_canvas, 'A. WRIST ANGLE: ', (downscaled_width + 50, 400), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, f'Right arm: {int(wrist_r)}', (downscaled_width + 50, 450), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    cv.putText(blank_canvas, f'Left arm: {int(wrist_l)}', (downscaled_width + 50, 500), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    cv.putText(blank_canvas, 'B. NECK: ', (downscaled_width + 50, 550), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, f'Neck angle: {int(neck_angle)}', (downscaled_width + 50, 600), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    cv.putText(blank_canvas, 'B. TRUNK: ', (downscaled_width + 50, 650), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv.putText(blank_canvas, f'Trunk angle: {int(trunk_angle)}', (downscaled_width + 50, 700), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    
    #Example usage of the RULA-calculator for left side
    upper_arm_score_l = rc.getUpperArmRULA(upper_arm_l , 0 , 0 , 0)
    lower_arm_score_l = rc.getLowerArmRULA(lower_arm_l , 0)
    wrist_score_l  = rc.getWristRULA(wrist_l , 0 , 1)
    neck_score = rc.getNeckRULA(neck_angle , 0 , 0)
    trunk_score = rc.getTrunkREBA(trunk_angle , 0, 0)
    leg_score_l = rc.getLegREBA(1)
    supp_l = 1
    print("Supp_l:",supp_l)

    #getting the RULA-table scores
    left_A_score = rc.get_Table_A_Score(upper_arm_score_l , lower_arm_score_l , wrist_score_l , supp_l)
    left_B_score = rc.get_Table_B_Score(neck_score , trunk_score , leg_score_l)

    #final score
    left_C_score = rc.get_Table_C_Score(left_A_score , left_B_score)
    print("RULA score for left side: " , left_C_score)


    #landmarks_tuple = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
    #reba_score = reba.getRebaScore(landmark_list)
    #print(reba_score)
    #cv.putText(blank_canvas, f'REBA score: {int(reba_score)} ,{reba_score}', ((downscaled_width +50, 50)), 
    #cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    print("UA:",upper_arm_l)
    print("LA:",upper_arm_r)
    cv.imshow('Ergonomics Assessment', blank_canvas)
    # Increment frame counter
    frame_count += 1   
    total_frames += 1
    # print(landmarks)

    # #write the output frame
    vid_writer.write(blank_canvas)

    ch = cv.waitKey(1)
    if ch == ord('q'):
        break
  
# cleanup the camera and close any open windows
cam.release()  
vid_writer.release()
cv.destroyAllWindows()
