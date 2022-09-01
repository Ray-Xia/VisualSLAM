from fileinput import filename
import os
from importlib_resources import path
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt 

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir):
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))# intrinsic parameters and projection matrix
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'poses.txt'))#ground truth poses
        self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.orb = cv2.ORB_create(3000)#设置要查找的最大关键点数量为3000

        # -- 使用FLANN快速最近邻搜索包里的匹配算法cv2.FlannBasedMatcher
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    # 什么时候需要用staticmethod? 
    # 当一个method不需要任何属于class的属性，但是这个method又应该属于这个classs时。
    @staticmethod #静态方法不需要实例化即可调用
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')#np.fromstring():从字符串的文本数据返回ndarray类型的一维数组; sep==seperate,字符串中的分隔符
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P # K is 3-by-3, P is 3-by-4 

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the Gound Truth poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The Gound Truth poses
        """
        poses = [] #poses是一个list对象
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1])) # vertically stack， T现在是4-by-4
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        img1 = self.images[i - 1]
        img2 = self.images[i]
      
        #keypoints1 = self.orb.detect(self.images[i - 1], None)
        keypoints1, descriptors1 = self.orb.detectAndCompute(img1, None)
        #keypoints2 = self.orb.detect(self.images[i], None)
        keypoints2, descriptors2 = self.orb.detectAndCompute(img2, None)



        ###########  把knnMatch替换成光流跟踪  ################ 
        q1 = np.array([ele.pt for ele in keypoints1],dtype='float32')
        lk_params = dict( winSize  = (21,21),
                    maxLevel = 7, # 最大金字塔层数
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        q2, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, q1, None, **lk_params)
        status = status.reshape(status.shape[0])
        ##find good one
        q1 = q1[status==1]
        q2 = q2[status==1]

        # draw_params = dict(matchColor = -1, # draw matches in green color
        #         singlePointColor = None,
        #         matchesMask = None, # draw only inliers
        #         flags = 2)

        # img3 = cv2.drawMatches(self.images[i], keypoints1, self. images[i-1],keypoints2, good ,None,**draw_params)
        # cv2.imshow("image", img3)
        # cv2.waitKey(0)
        # plt.imshow(img3, 'gray'),plt.show()
        # plt.imshow(self.images[i]),plt.show()
        # plt.imshow(self.images[i-1]),plt.show()

        return q1, q2

        # This function should detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        # The descriptors should then be matched using the class flann object (knnMatch with k=2)
        # Remove the matches not satisfying Lowe's ratio test
        # Return a list of the good matches for each image, sorted such that the n'th descriptor in image i matches the n'th descriptor in image i-1
        # https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html
        pass # pass is just a placeholder

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """

        Essential, mask = cv2.findEssentialMat(q1, q2, self.K) # method有RANSAC，LMEDS，默认RANSAC
        # print ("\nEssential matrix:\n" + str(Essential))

        R, t = self.decomp_essential_mat(Essential, q1, q2)

        return self._form_transf(R,t)

        # Estimate the Essential matrix using built in OpenCV function
        # Use decomp_essential_mat to decompose the Essential matrix into R and t
        # Use the provided function to convert R and t to a transformation matrix T
        pass

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """


        R1, R2, t = cv2.decomposeEssentialMat(E)
        T1 = self._form_transf(R1,np.ndarray.flatten(t))
        T2 = self._form_transf(R2,np.ndarray.flatten(t))
        T3 = self._form_transf(R1,np.ndarray.flatten(-t))
        T4 = self._form_transf(R2,np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        
        # Homogenize K
        K = np.concatenate(( self.K, np.zeros((3,1)) ), axis = 1) # K is 3-by-4 now, K = [K,0]

        # List of projections
        projections = [K @ T1, K @ T2, K @ T3, K @ T4] # @ means matrix multiplication

        np.set_printoptions(suppress=True) # suppress=True, always print floating point numbers using fixed point notation

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        positives = []
        for P, T in zip(projections, transformations): # projection维度4x3x4, transformations维度4x4x4
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T) # q1.T，q2.T是转秩
            hom_Q2 = T @ hom_Q1
            # Un-homogenize
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :] # 第四维归一化
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]  

            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0) # 统计Q1，Q2深度（Z）值大于0的个数
            relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)/
                                     np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)) #这是干啥？？
            positives.append(total_sum + relative_scale)
            

        # Decompose the Essential matrix using built in OpenCV function
        # Form the 4 possible transformation matrix T from R1, R2, and t
        # Create projection matrix using each T, and triangulate points hom_Q1
        # Transform hom_Q1 to second camera using T to create hom_Q2
        # Count how many points in hom_Q1 and hom_Q2 with positive z value
        # Return R and t pair which resulted in the most points with positive z

        max = np.argmax(positives) #返回最大值的索引
        # print("max = " + str(max))
        if (max == 2):
            # print(-t)
            return R1, np.ndarray.flatten(-t)
        elif (max == 3):
            # print(-t)
            return R2, np.ndarray.flatten(-t)
        elif (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t)
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t)



def main():
    data_dir = 'KITTI_sequence_2'  # Try KITTI_sequence_2 too
    vo = VisualOdometry(data_dir)


    play_trip(vo.images)  # Comment out to not play the trip

    gt_path = []
    estimated_path = []
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
        if i == 0:
            cur_pose = gt_pose
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))# 上一帧相机的位姿即要求解的位姿
            print ("\nGround truth pose:\n" + str(gt_pose))
            print ("\n Current pose:\n" + str(cur_pose))
            print ("The current pose used x,y: \n" + str(cur_pose[0,3]) + "   " + str(cur_pose[2,3]) )
        gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        
  
    
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir)+ "-optFlow"+ ".html")


if __name__ == "__main__":
    main()
