####### Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io as skio
from PIL import Image
import skimage.transform as sktr
import cv2
import os
import gc

###### Defining Essentials
total_frames = 60 # 4s of video @ 15 fps. We will extract in total 61 frames but will 
H = 224 # Resizing size of the Height Dimensions
W = 224 # Resizing size of the Width Dimensions

###### Frame Generation
def frame_gen(filepath,frame_num):

    """
    Function to Process Frames

    INPUTS:-
    1) filepath: Path to the Image Folder
    2) frame_num: The Index of the frame

    OUTPUTS:-
    1) op_frame: Output of frame Processing
    """

    ##### Image Path
    filepath_lsb = filepath+'/lsb'+'/lsb'+str(frame_num)+'.png'
    filepath_msb = filepath+'/msb'+'/msb'+str(frame_num)+'.png'

    ##### Image Reading
    #### LSB File
    img_lsb = cv2.cvtColor(cv2.imread(filepath_lsb), cv2.COLOR_BGR2GRAY)

    #### MSB File
    ### File-Reading
    img_msb = cv2.imread(filepath_msb)[:,:,0]

    ### File-Conversion
    img_msb_list = [] # List to store updated MSB Image
    
    for i in range(img_msb.shape[0]):

        img_msb_row_curr = [] # List to store values

        for j in range(img_msb.shape[1]):

            img_msb_curr = img_msb[i,j] # Current Image
            img_msb_curr_bin = np.binary_repr(img_msb_curr) # Binary Representation

            for k in range(8): # Number of Indexes to be added

                img_msb_curr_bin = img_msb_curr_bin + '0' # Appending 0s at the end to make it 16 bit

            img_msb_row_curr.append(int(img_msb_curr_bin,2))

        img_msb_list.append(img_msb_row_curr)

    img_msb = np.array(img_msb_list)

    #### LSB-MSB Combination
    op_frame = np.double(img_lsb) + np.double(img_msb)
    op_frame = (op_frame/256).astype('uint8')
    op_frame = op_frame[150:400,200:450] # Cropping the Frame Adequately
    op_frame = cv2.resize(op_frame, (224,224),  interpolation = cv2.INTER_NEAREST_EXACT) # Resizing the Ouptut Frame
    return op_frame

###### Background Substraction
def bg_substractor(frame,bg_frame,threshold):

    """
    Function to Substract Background

    INPUTS:-
    1) frame: Frame from which Background is to be substracted
    2) bg_frame: Background Frame
    3) threshold: Threshold for Foreground Mask

    OUTPUTS:-
    1) op_frame: Output of frame Processing
    """

    fg_mask = ((bg_frame - frame) > threshold) # Foreground Mask
    op_frame = np.multiply(frame,fg_mask) # Mask Application
    return op_frame

###### Optical Flow Estimation
def optical_flow_estimator(video_seq):

    """
    Estimating Franeback Flow for Video Sequence

    INPUTS:-
    1) video_seq: Input video sequence of shape (T,H,W) in this case

    OUTPUTS:-
    1) flow_op: Output Flow of the shape (T,H,W,2). 
                Here two channels correspond to Magnitude and Direction of the Flow
    """

    ##### Defining Essentials
    flow_op = [] # List to store Flow Outputs
    frames_total = int(video_seq.shape[0]) # A Count on total number of flows in the Video Sequence

    ##### Iterating till Tth from the first Frame
    for frame_idx in range(1,frames_total):

        frame_curr = video_seq[frame_idx] # Current Frame
        frame_prev = video_seq[frame_idx-1] # Previous Frame

        flow = cv2.calcOpticalFlowFarneback(frame_prev, frame_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0) # Optical Flow Extraction
        mag, dir = cv2.cartToPolar(flow[...,0],flow[...,1]) # Magnitude and Direction of Flow
        mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Scaling the Magnitude
        dir = dir*180/np.pi/2 # Scaling the Directional Flow
        flow = np.stack([mag,dir],axis=-1) # Stacking the Arrays
        flow_op.append(flow) # Appending the Flow

    return np.array(flow_op)

###### Symboitic Feature Maps
def symbiotic_extractor(video_seq):

    """
    Estimating Franeback Flow for Video Sequence

    INPUTS:-
    1) video_seq: Input video sequence of shape - (T,H,W) in this case

    OUTPUTS:-
    1) symb_op: Symbiotic Feature of shape - (T,H,W). 
    """
    ##### Defining Essentials
    symb_op = [] # List to store Sym Outputs
    frames_total = int(video_seq.shape[0]) # A Count on total number of flows in the Video Sequence

    ##### Iterating till Tth from the First Frame
    for frame_idx in range(1,frames_total):

        frame_curr = video_seq[frame_idx] # Current Frame
        frame_prev = video_seq[frame_idx-1] # Previous Frame
        frame_curr = frame_curr - frame_prev # Symbiotic Feature Extraction
        symb_op.append(frame_curr) # Appending it to the list

    return np.array(symb_op)

###### Gesture Sequence Generator
def gesture_seq_gen(folder_path,threshold):

    """
    Function to return a Gesture Sequence from the Folder Path

    INPUTS:-
    1) folder_path: Path to the Input Folder
    2) threshold: Background substraction threshold0

    OUTPUTS:-
    gesture_seq: Output Sequence of the Gesture
    """

    ##### Defining Essentials
    gesture_seq = [] # List to store frames
    bg_frame = frame_gen(folder_path,1) # Background Frame
    total_frames = len(os.listdir(folder_path+'./lsb')) # Computing Total Number of Frames 

    ##### Iterating over the Folder 
    for frame_idx in range(1,total_frames,2): # Dropping the First Frame, stepping@ 15fps instead of 30

        frame_curr = frame_gen(folder_path,frame_idx+1) # Getting the Current Frame
        frame_curr = bg_substractor(frame_curr,bg_frame,threshold) # Background Substraction
        gesture_seq.append(frame_curr)

    ##### Frame-Rate Adjustments
    total_frames_curr = len(gesture_seq) # Total Number of Frames in the Gesture Sequence

    if(total_frames_curr < 61): # Zero-Padding: If frames are less than 61 

        for f_rem in range(61 - len(gesture_seq)):
            frame_add = np.zeros((224,224),dtype=np.double) # Frame to be Added
            gesture_seq.append(frame_add)

        gesture_seq = np.array(gesture_seq)

    elif(len(gesture_seq) >= 61): # Z

        gesture_seq = np.array(gesture_seq)[:61] # Slicing: If frames are greater than or equal to 61

    return gesture_seq
     
##### Testing
#### Flow Extraction
folder_path = './Test007'
gesture_seq_op = gesture_seq_gen(folder_path,5)
gesture_flow_op = optical_flow_estimator(gesture_seq_op)
gesture_symb_op = symbiotic_extractor(gesture_seq_op)

#### Symbiotic Feature Visualization
for frame_idx in range(total_frames):

    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(gesture_seq_op[frame_idx])
    ax2.imshow(gesture_symb_op[frame_idx])
    plt.show()

#### Flow Visualization
#for frame_idx in range(total_frames): 

#    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
#    ax1.imshow(gesture_seq_op[frame_idx])
#    ax2.imshow(gesture_flow_op[frame_idx,:,:,0],cmap='gray')
#    ax3.imshow(gesture_flow_op[frame_idx,:,:,1],cmap='gray')
#    plt.show()

#op_frame = frame_gen(folder_path,75)
#bg_frame = frame_gen(folder_path,1)
#op_frame = bg_substractor(op_frame,bg_frame,5)
#print(np.max(op_frame),op_frame.shape)
#plt.imshow(op_frame,cmap='gray')
#plt.colorbar()
#plt.show()
