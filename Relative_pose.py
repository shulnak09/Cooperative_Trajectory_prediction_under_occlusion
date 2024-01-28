


import numpy as np
import matplotlib.pyplot as plt
import cv2
from numpy.linalg import matrix_rank
import math
import time
import os
import glob
from scipy.spatial.transform  import Rotation as Rot

'''
The function compute_pose() matches the features and computes the initial pose
between images as seen by two agents. It takes both the perspective images as well 
as the calibration matrices as inputs and returns the pose [R | t].
'''


def degeneracyCheckPass(first_points, second_points, rot, trans):
    rot_inv = rot
    for first, second in zip(first_points, second_points):
        first_z = np.dot(rot[0, :] - second[0] * rot[2, :], trans) / np.dot(rot[0, :] - second[0] * rot[2, :], second)
        first_3d_point = np.array([first[0] * first_z, second[0] * first_z, first_z])
        second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T, trans)

        if first_3d_point[2] < 0 or second_3d_point[2] < 0:
            return False

    return True




def compute_pose(img1, img2, K_1, K_2, scale=0.2):
    
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # Define the various Descriptors
    sift = cv2.SIFT_create(contrastThreshold= 0.04, sigma= 1.6)
    # surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()
    kaze = cv2.KAZE_create(nOctaveLayers= 3,extended= True)
    brisk = cv2.BRISK_create(octaves= 3, thresh = 30)
    fast = cv2.FastFeatureDetector_create()
    akaze = cv2.AKAZE_create(nOctaveLayers= 3)


    # starting time
    start = time.time()

    # find the keypoints and descriptors
    kp1,des1 = sift.detectAndCompute(img1, None)
    kp2,des2 = sift.detectAndCompute(img2, None)


    # img_kp1 = cv2.drawKeypoints(img1,kp1,img1, color= (0,255,0))
    # cv2.imwrite('Keypoints_First_cam.jpg', img_kp1)

    # img_kp2 = cv2.drawKeypoints(img2,kp2,img2, color= (0,255,0))
    # cv2.imwrite('Keypoints_second_cam.jpg', img_kp2)


    

    my_feature = "sift"

    if my_feature =="sift" or my_feature =="kaze":
        # Matcher for String and  Descriptors (SIFT, SURF, KAZE):
        # BFMatcher with default params
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
        # Match Descriptors
        matches = bf.knnMatch(des1, des2, k=2)
        # Find the good matches using Ratio Test
        matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

    else:
        # Matcher for Binary Descriptor (ORB, BRISK, AKAZE):
        # create BFMatcher object with Hamming distance

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck= False)
        # Match descriptors.
        matches = bf.knnMatch(des1, des2, k=2)
        # Find the good matches using Ratio Test
        matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

    print("Total Matches: %d" %len(matches))



    # Feature Matching time
    feature_time = time.time()

 
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append([m])


    MIN_MATCH_COUNT = 20


    if len(matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches])



        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC,2,  0.99)
        matches = [m for m, v in zip(matches, mask.ravel()) if v]



    else:
        print("Not enough matches are found - %d/%d" % (len(matches),MIN_MATCH_COUNT))
        matchesMask = None

    print("Total Matches after Masking: %d" %len(matches))


    # converting source and destination points into numpy array and saving them
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)


    # Draw first 10 matches.
    matchesMask = [1 if m is not None  else 0 for m in matches]
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,**draw_params)
    cv2.imshow("Matched Image", img3)
    cv2.imwrite('Matched_image_sift.jpg', img3)


    # Feature Matching time
    end_time = time.time()
    
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    
    # Total Matched Feaures:
    print("KP1: %d, KP2: %d, AVG: %d" %(len(kp1),len(kp2), 0.5*(len(kp1)+len(kp2))))

    
    # total time taken for matching features:
    print(f"Runtime for feature matching:  {feature_time - start}")


    # total time taken for calculating Fundamental matrix:
    print(f"Runtime for program:  {end_time - start}")

    
    # Calculate the Essential Matrix
    E = K_1.T.dot(F).dot(K_2)
    # print(E)



    def decomposeEsssentialMatrix(E) : 
        U, S, Vt = np.linalg.svd(E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape(3, 3)
        K_inv_1 = np.linalg.inv(K_1)
        K_inv_2 = np.linalg.inv(K_2)
        first_inliers = []
        second_inliers = []
        for i in range(len(src_pts)):
            first_inliers.append(K_inv_1.dot([src_pts[i][0], src_pts[i][1], 1.0]))
            second_inliers.append(K_inv_2.dot([dst_pts[i][0], dst_pts[i][1], 1.0]))

        # First choice: R = U * W * Vt, T = u_3
        R_new = U.dot(W).dot(Vt)
        T_new = U[:, 2]

        # Start degeneracy checks
        if not degeneracyCheckPass(first_inliers, second_inliers, R_new, T_new):
            # Second choice: R = U * W * Vt, T = -u_3
            T_new = - U[:, 2]
            if not degeneracyCheckPass(first_inliers, second_inliers, R_new, T_new):
                # Third choice: R = U * Wt * Vt, T = u_3
                R_new = U.dot(W.T).dot(Vt)
                T_new = U[:, 2]
                if not degeneracyCheckPass(first_inliers, second_inliers, R_new, T_new):
                    # Fourth choice: R = U * Wt * Vt, T = -u_3
                    T_new = - U[:, 2]
        
        return R_new, T_new

    [R_new, T_new] = decomposeEsssentialMatrix(E)
    # print("New Rotation Matrix",R_new)
    # print("New Translation vector",T_new)
    
    r = Rot.from_matrix(R_new)
    Rot2Eul  = r.as_euler('yxz', degrees = True) # y=  yaw, x = pitch,  z = roll 
    print("Rotation Matrix in Degrees :{}".format(Rot2Eul))
    print("Translation vector [t]: {}".format(T_new))

    
    # Second method for [R | t]:
    [R1, R2, t] = cv2.decomposeEssentialMat(E)
    print("First Rotation Matrix",R1)
    print("Second Rotation Matrix",R2)
    print("Translation vector",t)
    R = R1 if abs(np.sum(np.diag(np.diag(R1)) - np.identity(3))) < abs(np.sum(np.diag(np.diag(R2))  - np.identity(3))) else R2

    # d_estimated = np.linalg.norm(t)
    # d_true = np.linalg.norm(0.0254*np.array([45.5, 5.5, 2.5]))

    # scale = d_true/d_estimated
    print("scale",scale)
    t *= scale 
    print("scaled Translation vector", t)
    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(R) :
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype = R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(R) :

        assert(isRotationMatrix(R))

        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    p = rotationMatrixToEulerAngles(R)
    eul_x = p[0] * (180/ math.pi)
    eul_y = p[1] * (180/ math.pi)
    eul_z = p[2] * (180/ math.pi)

    Rot2Eul_1 = np.array([eul_x, eul_y, eul_z])
    print("Pitch: %.3f, Roll : %.3f, Yaw: %.3f" % (eul_z, eul_x , eul_y))


    return R_new,Rot2Eul, T_new, R, Rot2Eul_1, t