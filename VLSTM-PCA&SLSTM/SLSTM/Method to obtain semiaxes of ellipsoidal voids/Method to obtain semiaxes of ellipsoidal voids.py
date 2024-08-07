'''
Method to obtain semiaxes of ellipsoidal voids
'''
import os
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
import time
# Setting the work directory
dir1 = 'D:\\ABAQUS_wxj\\RVE_6061AL\\all data\\ell\\x0y0z0_168void_whole_eq_AL'
dir2 = 'D:\\pycharm_file\\3.AI-AL\\x0y0z0_100fra_CH_cubic_w20'  # save directory
os.chdir(dir1)
num_frames = 101
T_max = 200
T_min = 33
num_eqstrain_divisions = 101
interpolation_method = 'cubic'
magnitude = len(str(num_frames))
T_magnitude = len(str(T_max))
'''
load file
'''
Eqstrain_data_all = []
void_data_all = []
for void in range(T_min, T_max+1):
    Eqstrain_data = pd.read_csv('./void-ell02T' + str(void).zfill(T_magnitude) + '/Code_file/void-Eqstrain.csv', header=None).iloc[1:, :]
    Eqstrain_data_all.append(Eqstrain_data)
    void_data_3d = []
    for i in range(num_frames):
        void_coord = pd.read_csv('./void-ell02T'+str(void).zfill(T_magnitude)+'/corner_node/COORD_corner'+str(i).zfill(magnitude) + '.csv',header=None).iloc[1:, 1:]  # 读取数据三个坐标
        mises_data = pd.read_csv('./void-ell02T' + str(void).zfill(T_magnitude) + '/corner_node/Mises_cornerpoint_' + str(i).zfill(magnitude) + '.csv', header=None).iloc[:, 1:]
        void_data_2d = np.hstack((void_coord, mises_data))
        void_data_3d.append(void_data_2d)
        print('void:', void, 'frame',i,'finish')
    void_data_all.append(void_data_3d)
Eqstrain_data_all = np.array(Eqstrain_data_all)
Eqstrain_data_all = Eqstrain_data_all.astype(np.float64)
void_data_all = np.array(void_data_all)
void_data_all = void_data_all.astype(np.float64)


# data save
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder create  ---")
    else:
        print("---  There is this folder!  ---")


mkdir(dir2)
os.chdir(dir2)
# data calculation
eqstrain_division_points = np.zeros((Eqstrain_data_all.shape[0],num_eqstrain_divisions))
for i in range(Eqstrain_data_all.shape[0]):
    eqstrain_division_points[i] = np.linspace(np.min(Eqstrain_data_all[i,:,13]), np.max(Eqstrain_data_all[i,:,13]), num_eqstrain_divisions)
np.save('./eqstrain_division_points_AL.npy', eqstrain_division_points)

VolumeProfile_division_points = np.zeros((Eqstrain_data_all.shape[0],num_eqstrain_divisions))
for i in range(len(Eqstrain_data_all)):
    interp_func = interpolate.interp1d(Eqstrain_data_all[i,:,13].flatten(), Eqstrain_data_all[i,:,3].flatten(), kind=interpolation_method)
    VolumeProfile_division_points[i] = interp_func(eqstrain_division_points[i])
np.save('./VolumeProfile_division_points_AL.npy', VolumeProfile_division_points)

Vfraction_division_points = np.zeros((Eqstrain_data_all.shape[0],num_eqstrain_divisions))
for i in range(len(Eqstrain_data_all)):
    interp_func = interpolate.interp1d(Eqstrain_data_all[i,:,13].flatten(), Eqstrain_data_all[i,:,8].flatten(), kind=interpolation_method)
    Vfraction_division_points[i] = interp_func(eqstrain_division_points[i])
np.save('./Vfraction_division_points_AL.npy', Vfraction_division_points)

VolumeSphere_division_points = np.zeros((Eqstrain_data_all.shape[0],num_eqstrain_divisions))
for i in range(len(Eqstrain_data_all)):
    interp_func = interpolate.interp1d(Eqstrain_data_all[i,:,13].flatten(), Eqstrain_data_all[i,:,5].flatten(), kind=interpolation_method)
    VolumeSphere_division_points[i] = interp_func(eqstrain_division_points[i])
np.save('./VolumeSphere_division_points_AL.npy', VolumeSphere_division_points/VolumeSphere_division_points[0][0])

void_data_all_cal = np.zeros((void_data_all.shape[0],num_eqstrain_divisions, void_data_all.shape[2], void_data_all.shape[3]))

for i in range(void_data_all.shape[0]):
    for j in range(void_data_all.shape[2]):
        for k in range(void_data_all.shape[3]):
            interpolator = interpolate.interp1d(Eqstrain_data_all[i,:,13].flatten(), void_data_all[i,:, j, k], kind=interpolation_method, fill_value="extrapolate")
            void_data_all_cal[i,:, j, k] = interpolator(eqstrain_division_points[i])

print("New void_data_all shape:", void_data_all_cal.shape)
np.save('./void_data_all_cal_AL.npy', void_data_all_cal)

'''
4methods
'''

# method1-----------------------------------------------------------
t0 = time.time()
eq_polar_axis_all = []
SMA_err_all = []
for i in range(void_data_all_cal.shape[0]):
    print('void', i, 'finish')
    eq_polar_axis = []
    SMA_err = []
    for j in range(void_data_all_cal.shape[1]):
        max_SMA_A = np.max(void_data_all_cal[i, j, :, 0], axis=0)
        min_SMA_A = np.min(void_data_all_cal[i, j, :, 0], axis=0)
        SMA_A = (np.abs(max_SMA_A) + np.abs(min_SMA_A)) / 2
        err_SMA_A = np.abs((np.abs(max_SMA_A) - np.abs(min_SMA_A))) / SMA_A

        max_SMA_B = np.max(void_data_all_cal[i, j, :, 1], axis=0)
        min_SMA_B = np.min(void_data_all_cal[i, j, :, 1], axis=0)
        SMA_B = (np.abs(max_SMA_B) + np.abs(min_SMA_B)) / 2
        err_SMA_B = np.abs((np.abs(max_SMA_B) - np.abs(min_SMA_B))) / SMA_B

        max_SMA_C = np.max(void_data_all_cal[i, j, :, 2], axis=0)
        min_SMA_C = np.min(void_data_all_cal[i, j, :, 2], axis=0)
        SMA_C = (np.abs(max_SMA_C) + np.abs(min_SMA_C)) / 2
        err_SMA_C = np.abs((np.abs(max_SMA_C) - np.abs(min_SMA_C))) / SMA_C

        eq_polar_axis.append([SMA_A, SMA_C, SMA_B])
        SMA_err.append([err_SMA_A, err_SMA_C, err_SMA_B])

    eq_polar_axis_all.append(eq_polar_axis)
    SMA_err_all.append(SMA_err)
eq_polar_axis_all = np.array(eq_polar_axis_all)
SMA_err_all = np.array(SMA_err_all)
np.save('./eq_polar_axis_all_long_distance.npy', eq_polar_axis_all)
t1 = time.time()
t_Longest_distance = t1-t0

# Method3-----------------------------------------------------------
eq_polar_axis_all1 = np.zeros((void_data_all.shape[0], num_eqstrain_divisions, 3))
for i in range(eq_polar_axis_all1.shape[0]):
    print('void', i, 'finish')
    for j in range(num_eqstrain_divisions):
        nodes = void_data_all_cal[i, j, :, :3]
        cov_matrix = np.cov(nodes, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        lengths = np.sqrt(eigenvalues)
        eq_polar_axis_all1[i, j] = lengths

np.save('./eq_polar_axis_all_characteristic_value.npy', eq_polar_axis_all1)
t2 = time.time()
t_characteristic_value = t2-t1

# Method4-----------------------------------------------------------
eq_polar_axis_all2 = np.zeros((void_data_all.shape[0], num_eqstrain_divisions, 3))
for i in range(eq_polar_axis_all2.shape[0]):
    print('void', i, 'finish')
    for j in range(num_eqstrain_divisions):
        coordinates = void_data_all_cal[i, j, :, :3]
        pca = PCA(n_components=3)
        pca.fit(coordinates)
        eq_polar_axis_all2[i, j] = np.sqrt(pca.explained_variance_)

np.save('./eq_polar_axis_all_PCA.npy', eq_polar_axis_all2)
t3 = time.time()
t_PCA = t3-t2

# Method4-----------------------------------------------------------


def ellipsoid_model(params, coords):
    a, b, c = params
    x, y, z = coords.T
    return ((x)**2 / a**2) + ((y)**2 / b**2) + ((z)**2 / c**2) - 1.0


# loss
def loss_function(params, coords):
    return np.sum(ellipsoid_model(params, coords)**2)


void_data_all_cal = np.load(file='./void_data_all_cal_AL.npy')
eq_polar_axis_all = []
for i in range(void_data_all_cal.shape[0]):
    eq_polar_axis = []
    print('void',i,'finish')
    for j in range(void_data_all_cal.shape[1]):
        eq_void_coord = void_data_all_cal[i,j,:,:3]
        initial_guess = np.array([0.5, 0.5, 0.5])
        # minimize
        result = minimize(loss_function, initial_guess, args=(eq_void_coord,), method='Powell')  #Powell/TNC
        a_fit, b_fit, c_fit = result.x
        eq_polar_axis.append([abs(a_fit),abs(b_fit),abs(c_fit)])
    eq_polar_axis_all.append(eq_polar_axis)

eq_polar_axis_all = np.array(eq_polar_axis_all)
np.save('./eq_polar_axis_all.npy', eq_polar_axis_all)
t4 = time.time()
t_convex_hull = t4-t3
# time

print('t_Longest_distance=',t_Longest_distance)
print('t_characteristic_value=',t_characteristic_value)
print('t_PCA=',t_PCA)
print('t_convex_hull=',t_convex_hull)

