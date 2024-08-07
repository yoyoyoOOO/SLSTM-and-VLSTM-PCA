'''
Voronoi tessellation for inverse operator
'''
import time
import os
import numpy as np
from scipy.interpolate import griddata

# # Setting the work directory
dir1 = r'D:\pycharm_file\3.AI-AL\x0y0z0_100fra_CH_cubic_w20'
Voronoi_meshnum = 100  # voronoi mesh
delete = 2
os.chdir(dir1)
void_data_all_cal = np.load(r'void_data_all_cal_AL.npy')


if delete == 2:
    plane = 'xy'
elif delete == 0:
    plane = 'yz'
elif delete == 1:
    plane = 'xz'


def void_voronoi(void_data, Voronoi_meshnum=100):
    frame = void_data.shape[1]
    void_data = void_data.reshape(-1, void_data.shape[2], void_data.shape[3])
    N = Voronoi_meshnum
    count = 0
    datasize = void_data.shape[0]
    vor_field_data = np.zeros((datasize, N, N),dtype=np.float32)
    global Xmax_void
    global Xmin_void
    global Ymax_void
    global Ymin_void
    Xmax_void, Xmin_void = void_data[:, :, 0].max(), void_data[:, :, 0].min()
    Ymax_void, Ymin_void = void_data[:, :, 1].max(), void_data[:, :, 1].min()
    for i in range(datasize):
        print(round(i/frame,0), 'void_voronoi begin')
        Xi, Yi, Zi = np.array(void_data[i, :, 0]), np.array(void_data[i, :, 1]), np.array(void_data[i, :, 2])
        Pi = np.zeros((void_data.shape[1], 2))
        Pi[:, 0] = Xi
        Pi[:, 1] = Yi
        x = np.linspace(Xmin_void, Xmax_void, N)
        y = np.linspace(Ymin_void, Ymax_void, N)
        X, Y = np.meshgrid(x, y)
        P = np.array([X.flatten(), Y.flatten()]).transpose()
        Z_nearest = griddata(Pi, Zi, P, method='linear')
        Z_nearest = Z_nearest.reshape([N, N])
        vor_field_data[count, :, :] = Z_nearest
        count += 1
    return vor_field_data


os.chdir(dir1)
void_num = void_data_all_cal.shape[0]
void_frame = void_data_all_cal.shape[1]
void_data = np.delete(void_data_all_cal, delete, axis=3)
t0 = time.time()
vor_data0 = void_voronoi(void_data, Voronoi_meshnum=Voronoi_meshnum)
t1 = time.time()
vor_time = t1-t0
print('vor_time=', vor_time)
vor_data = np.reshape(vor_data0,(void_data_all_cal.shape[0], -1, vor_data0.shape[1],vor_data0.shape[2]))
np.save('voronoi_coord_Mises_'+str(Voronoi_meshnum)+'mesh_AL_'+plane+'.npy',vor_data)


# voronoi to scatter----------------------------------------
def inverse_voronoi(vor_data, Xmin, Xmax, Ymin, Ymax):
    void = len(vor_data)
    vor_data = np.reshape(vor_data,(-1,vor_data.shape[2],vor_data.shape[3]))
    datasize, N1, N2 = vor_data.shape
    void_data = np.zeros((datasize, N1 * N2, 3))
    for i in range(datasize):
        Z = vor_data[i, :, :].flatten()
        all_indices = np.arange(len(Z))
        X, Y = np.meshgrid(np.linspace(Xmin, Xmax, N1), np.linspace(Ymin, Ymax, N2))
        X = X.flatten()[all_indices]
        Y = Y.flatten()[all_indices]
        Z = Z[all_indices]
        void_data[i, :, 0] = X
        void_data[i, :, 1] = Y
        void_data[i, :, 2] = Z
    void_data = np.reshape(void_data,(void,-1,N1*N2,3))
    return void_data


inverse_data = inverse_voronoi(vor_data, Xmin_void, Xmax_void, Ymin_void, Ymax_void)
np.save('voronoi_inverse_data_'+str(Voronoi_meshnum)+'mesh_AL_'+plane+'.npy',inverse_data)
