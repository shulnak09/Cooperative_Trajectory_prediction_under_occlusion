# Import Libraries:
import pyzed.sl as sl
import numpy as np
import cv2
import pickle as pkl
import time
from PIL import Image, ImageDraw, ImageFont
import time
from Relative_pose import compute_pose, degeneracyCheckPass
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


def interpolate_nan(initial_array):
    t = np.linspace(0.4, 8, 20) # Time steps for collecting 
    # Perform cubic spline interpolation for each column
    for col in range(initial_array.shape[1]):
        nan_indices = np.isnan(initial_array[:, col])
        
        if np.any(nan_indices):
            t_interp = t[~nan_indices]
            y_interp = initial_array[~nan_indices, col]
            cs = CubicSpline(t_interp, y_interp)
            initial_array[nan_indices, col] = cs(t[nan_indices])

# open the two zed cameras:
zed = [sl.Camera(), sl.Camera()]

# Initialize parameters for the cameras
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD1080 video mode
init_params.camera_fps = 30 # Set fps at 30
init_params.sdk_verbose = True # Enable the verbose mode
init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Set the depth mode to performance (fastest)
init_params.coordinate_units = sl.UNIT.METER  # Use milimeter units (for depth measurements)

# Initialize the zed camera with two 
zed[0].open(init_params)
zed[1].open(init_params)

# Object detection class for ZED
obj_param = sl.ObjectDetectionParameters()


obj_param.enable_tracking = True
obj_param.image_sync = True
# obj_param.enable_mask_output = False
obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_ACCURATE


# Obtain the camera Calibration Parameters:
cam1_info = zed[0].get_camera_information().camera_configuration.calibration_parameters.left_cam
cam2_info = zed[1].get_camera_information().camera_configuration.calibration_parameters.left_cam

# Obtain the camera intrinsics:
## Camera 1 Parameters ##
[fx,fy] = cam1_info.fx, cam1_info.fy
[cx,cy] = cam1_info.cx, cam1_info.cy
K_1 = np.array([[fx, 0 , cx ],
                [0 , fy, cy],
                [0, 0, 1]])
K_inv_1 = np.linalg.inv(K_1)

## Camera 2 Parameters ##
[fx,fy] = cam2_info.fx, cam2_info.fy
[cx,cy] = cam2_info.cx, cam2_info.cy
K_2 = np.array([[fx, 0 , cx ],
                [0 , fy, cy],
                [0, 0, 1]])
K_inv_2 = np.linalg.inv(K_2)
##

'''
ZED SDK provides undistorted rectified image so no need to undistort the image 
using distortion coefficient. 
'''

## Camera Grab for image pair:
img1 = sl.Mat(1280, 720, sl.MAT_TYPE.U8_C4)
img2 = sl.Mat(1280, 720, sl.MAT_TYPE.U8_C4)
# img1 = sl.Mat()
# img2 = sl.Mat()

# Camera depth information:
depth1 = sl.Mat()
depth2 = sl.Mat()



# Enable Positional Tracking of the camera:

positional_tracking_param = sl.PositionalTrackingParameters()
positional_tracking_param.set_floor_as_origin = True

zed[0].enable_positional_tracking(positional_tracking_param)
zed[0].enable_object_detection(obj_param)

zed[1].enable_positional_tracking(positional_tracking_param)
zed[1].enable_object_detection(obj_param)



objects = [sl.Objects(), sl.Objects()]

obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
obj_runtime_param.detection_confidence_threshold = 25
runtime_parameters = sl.RuntimeParameters()

train_traj_cam_1 = []
train_traj_cam_2 = []
train_traj_cam_1_transf = []

print("Trajectory Tracking Started")
i = 0
Num_frames = 264 # at 30 FPS i.e. 8.8 seconds
while i < Num_frames:

    # Ego Agent:
    zed[0].grab()
    zed[0].retrieve_image(img1, sl.VIEW.LEFT)
    zed[0].retrieve_measure(depth1, sl.MEASURE.XYZRGBA)
    img1_color = img1.get_data()
    timestamp1 = zed[0].get_timestamp(sl.TIME_REFERENCE.CURRENT)

    # Static Camera:
    zed[1].grab()
    zed[1].retrieve_image(img2, sl.VIEW.LEFT)
    zed[1].retrieve_measure(depth2, sl.MEASURE.XYZRGBA)
    img2_color = img2.get_data()
    timestamp2 = zed[1].get_timestamp(sl.TIME_REFERENCE.CURRENT)

    
    zed[0].retrieve_objects(objects[0], obj_runtime_param)
    zed[1].retrieve_objects(objects[1], obj_runtime_param)
    # obj_array2 = objects[1].object_list
    
    if objects[0].is_new:
        obj_array1 = objects[0].object_list
        if len(obj_array1) > 0 :
            first_object = obj_array1[0]
            position = first_object.position
            velocity = first_object.velocity
            dimensions = first_object.dimensions
            if first_object.mask.is_init():
                print(" 2D mask available")
            
            bounding_box_2d = first_object.bounding_box_2d
            bbox = []
            for it in bounding_box_2d :
                bbox.append(it)
            bbox = np.array(bbox)
            bbox_center = (int((bbox[0][0] + int (bbox[1][0]))/2),int((bbox[0][1] + int (bbox[3][1]))/2)) 
            point3D_zed0 = depth1.get_value(bbox_center[0],bbox_center[1])
            point3D_zed0 = np.array(point3D_zed0[1])
            
            bounding_box = first_object.bounding_box
            np.set_printoptions(precision=3)
            cv2.rectangle(img1_color, bounding_box_2d[0,:].astype(int),bounding_box_2d[2,:].astype(int), (0,0,255),2)
            cv2.putText(img1_color, 
                        text = str(position), 
                        org = bbox_center, 
                        fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 1.0,
                        color = (125, 246, 55),
                        thickness = 1)
            if i % 12 == 0 and i >24:
                
                ped_coord = ([point3D_zed0[0],  point3D_zed0[2], velocity[0], velocity[2]])
                train_traj_cam_2.append(ped_coord)
                
    if i % 12 == 0 and i !=0:
        filepath = "./frames_goodwin/TIV_results/CAM_2/frame_{}.jpg".format(i)
        cv2.imwrite(filepath, img1_color)
        
            
    
    if objects[1].is_new:
        obj_array2 = objects[1].object_list
        if len(obj_array2) > 0 :
            first_object = obj_array2[0]
            position = first_object.position
            # print("Position: {}".format(position))
            velocity = first_object.velocity
            dimensions = first_object.dimensions
            if first_object.mask.is_init():
                print(" 2D mask available")
            
            bounding_box_2d = first_object.bounding_box_2d
            bbox = []
            for it in bounding_box_2d :
                bbox.append(it)
            bbox = np.array(bbox)
            bbox_center = (int((bbox[0][0] + int (bbox[1][0]))/2),int((bbox[0][1] + int (bbox[3][1]))/2)) 
            point3D_zed1 = depth2.get_value(bbox_center[0],bbox_center[1])
            point3D_zed1 = np.array(point3D_zed1[1])
            # print("3D_point_cloud: {}".format(point3D_zed1))
            bounding_box = first_object.bounding_box
            np.set_printoptions(precision=3)
            cv2.rectangle(img2_color, bounding_box_2d[0,:].astype(int),bounding_box_2d[2,:].astype(int), (0,0,255),2)
            cv2.putText(img2_color, 
                        text = str(position), 
                        org = bbox_center, 
                        fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 1.0,
                        color = (125, 246, 55),
                        thickness = 1)
            if i % 12 == 0 and i !=0:
                
                ped_coord = ([point3D_zed1[0],  point3D_zed1[2], velocity[0], velocity[2]])
                ped_coord_aug = np.array(ped_coord)
                ped_coord_aug = np.insert(ped_coord_aug, 1, 0)
                ped_coord_aug = np.insert(ped_coord_aug, 4, 0)
                # print(ped_coord_aug)
                # print("Pedestrian [X Y Z]",ped_coord_aug[:,:3])
                coord_transf = ( [np.squeeze(np.matmul(R.T,(ped_coord_aug[:3].reshape(-1,1)-t))), 
                                  np.squeeze(np.matmul(R.T,(ped_coord_aug[3:].reshape(-1,1)))) ])
                train_traj_cam_1.append(ped_coord)
                coord_transf = np.delete(coord_transf, [1,4])
                train_traj_cam_1_transf.append(coord_transf)


            
            if i == 6:
                # Pose Recovery: Use R,t for now
                # scale = np.linalg.norm(point3D_zed0[:3]-point3D_zed1[:3])
                [R_new,Rot2Eul, T_new, R, Rot2Eul_1, t] = compute_pose(img1_color, img2_color, K_1, K_2, scale =1.237)
                # time.sleep(1) # Pause code
    if i % 12 == 0 and i >24:            
        filepath = "./frames_goodwin/TIV_results/CAM_1/frame_{}.jpg".format(i)
        cv2.imwrite(filepath, img2_color)

    i += 1

    cv2.imshow("cam1", img1_color)
    cv2.imshow("cam2", img2_color)


    key = cv2.waitKey(1)

train_traj_cam_1 = np.array(train_traj_cam_1)
train_traj_cam_1  =train_traj_cam_1[1:,:]
train_traj_cam_1[np.isnan(train_traj_cam_1)] =0

train_traj_cam_1_transf = np.array(train_traj_cam_1_transf)
train_traj_cam_1_transf  =train_traj_cam_1_transf[1:,:]
train_traj_cam_1_transf[np.isnan(train_traj_cam_1_transf)] =0

train_traj_cam_2 = np.array(train_traj_cam_2)
train_traj_cam_2  =train_traj_cam_2[1:,:]
train_traj_cam_2[np.isnan(train_traj_cam_2)] =0

np.set_printoptions(precision = 3)
print("trajectory one :", train_traj_cam_1 )
print("trajectory one transformed", train_traj_cam_1_transf)
print("trajectory two :", train_traj_cam_2)

traj_name = 'turning_left'

with open(f'./frames_goodwin/TIV_results/CAM_1/{traj_name}.pkl','wb') as f:
    pkl.dump(train_traj_cam_1, f)

with open(f'./frames_goodwin/TIV_results/CAM_1/{traj_name}_transf.pkl','wb') as f:
    pkl.dump(train_traj_cam_1_transf, f)
    
with open(f'./frames_goodwin/TIV_results/CAM_2/{traj_name}.pkl','wb') as f:
    pkl.dump(train_traj_cam_2, f)
    
with open(f'./frames_goodwin/TIV_results/CAM_2/{traj_name}_params.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pkl.dump([R, t], f)

# close the camera
zed[0].close()
zed[1].close()


fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Extract the x and y coordinates for each modified array
x1, y1 = train_traj_cam_1[:, 0] , train_traj_cam_1[:, 1] 
x2, y2 = train_traj_cam_1_transf[:, 0]  , train_traj_cam_1_transf[:, 1] 
x3, y3 = train_traj_cam_2[:, 0]  , train_traj_cam_2[:, 1] 
# x4, y4 =  x2 - x3, y2 - y3
# error = np.array([[x4],[y4]])
# error = np.linalg.norm(error)

# Scatter plots for each modified array
axs[0, 0].scatter(x1, y1, marker='o', label='static')  # Use 'o' marker for modified array 1
axs[0, 1].scatter(x2, y2, marker='s', label='static_transf')  # Use 's' marker for modified array 2
axs[1, 0].scatter(x3, y3, marker='^', label='ego')  # Use '^' marker for modified array 3
# axs[1, 1].scatter(x4, y4, marker='d', label='error')

# Set aspect ratio to 'equal' for all subplots
axs[0, 0].set_aspect('equal', adjustable='box')
axs[0, 1].set_aspect('equal', adjustable='box')
axs[1, 0].set_aspect('equal', adjustable='box')
# axs[1, 1].set_aspect('equal', adjustable='box')

# Add legends to each subplot
axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
# axs[1, 1].legend()

# plt.legend('Camera_1','Camera_1_transf','Camera_2')
# Defining custom 'xlim' and 'ylim' values.
custom_xlim = (-3, 3)
custom_ylim = (-1, 7)

# Setting the values for all axes.
plt.setp(axs, xlim=custom_xlim, ylim=custom_ylim)

# Adjust spacing between subplots
plt.tight_layout()

# plt.savefig('./frames_goodwin/intermittent_occlusion/CAM_1/traj_occ3.png')

# Show the plot
plt.show()

