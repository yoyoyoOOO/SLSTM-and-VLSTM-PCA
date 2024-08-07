import os
from numpy import array
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽输出warning
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from random import shuffle
import time
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam
from scipy.spatial import ConvexHull

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Setting the work directory------------------------------------------
dir_parameters = 'x0y0z0_100fra_CH_cubic_w20'
dir1 = 'D:\\pycharm_file\\3.AI-AL\\'
dir2 = 'D:\\pycharm_file\\3.AI-AL\\VLSTM-PCA network structure'
voronoi_mesh = 100
n_component = 50  # PCA
# LSTM
n_steps_in, n_steps_out, n_features, overlap = 5, 5, n_component, 0
ratio_split = 0.8
rms = RMSprop(learning_rate=0.001, decay=0.0001)
adam = Adam(learning_rate=0.001, decay=0.0001)
batch_size = 16
epochs = 100
units = 200

down_lim = 0
decrease_frame = 1
# name
model_number = 2
name_index = dir_parameters.find("_")  # “x0y0z0”
w = dir_parameters[-3:]
model_name = str(model_number) + '_' + dir_parameters[:name_index] + '_' + str(168 - down_lim) + 'void_' + w
delete = 2

# load data--------------------------------------------------------------
os.chdir(dir1 + dir_parameters)
void_data_all_cal = np.load(r'void_data_all_cal_AL.npy')
eqstrain_division_points = np.load(file='./eqstrain_division_points_AL.npy')
vor_data = np.load('voronoi_coord_Mises_' + str(voronoi_mesh) + 'mesh_AL_xy.npy')
void_num = void_data_all_cal.shape[0]
void_frame = void_data_all_cal.shape[1] - decrease_frame


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder create  ---")
    else:
        print("---  There is this folder!  ---")


def save_number_to_txt(number, filename):
    with open(filename, 'w') as file:
        file.write(str(number))


mkdir(dir2)
os.chdir(dir2)
# pca--------------------------------------------------------------------------------
pca = PCA(n_components=n_component)
vor_data_pca = pca.fit_transform(vor_data)
variance_ratio = pca.explained_variance_ratio_
variance_ratio_sum = np.sum(variance_ratio)

filename = 'PCA_variance_ratio_sum.txt'
save_number_to_txt(variance_ratio_sum, filename)
print('PCA variance_ratio_sum', variance_ratio_sum)
vor_data_pca_map = np.zeros_like(vor_data_pca)
# Scale
for component in range(n_component):
    data_flatten = vor_data_pca[:, component:component + 1]
    scaler_data_pca = MinMaxScaler(feature_range=(-1, 1))
    scaler_data_pca.fit(data_flatten)
    data_flatten_map = scaler_data_pca.transform(data_flatten)
    vor_data_pca_map[:, component:component + 1] = data_flatten_map

vor_data_pca_map = np.reshape(vor_data_pca_map, (void_num, void_frame, -1))

#  Random disruption of data
Ind = list(range(len(vor_data_pca_map)))
shuffle(Ind)
Ind_train = Ind[0:round(ratio_split * len(vor_data_pca_map))]
Ind_test = Ind[round(ratio_split * len(vor_data_pca_map)):]


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out, overlap):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out - overlap
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix - overlap:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


vor_data_pca_map_train = vor_data_pca_map[Ind_train]
vor_data_pca_map_test = vor_data_pca_map[Ind_test]

# training dataset
X_train = np.empty((0, n_steps_in, n_features))
y_train = np.empty((0, n_steps_in, n_features))
for i in range(len(vor_data_pca_map_train)):
    X_coord_eq, y_coord_eq = split_sequences(vor_data_pca_map_train[i], n_steps_in, n_steps_out,overlap)
    X_train = np.concatenate((X_train, X_coord_eq), axis=0)
    y_train = np.concatenate((y_train, y_coord_eq), axis=0)
# testing dataset
X_test = np.empty((0, n_steps_in, n_features))
y_test = np.empty((0, n_steps_in, n_features))
for i in range(len(vor_data_pca_map_test)):
    X_coord_eq, y_coord_eq = split_sequences(vor_data_pca_map_test[i], n_steps_in, n_steps_out, overlap)
    X_test = np.concatenate((X_test, X_coord_eq), axis=0)
    y_test = np.concatenate((y_test, y_coord_eq), axis=0)

# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------
df_test = pd.DataFrame(Ind_test)
df_test.to_csv(str(model_number) + '_Ind_test.csv', index=False, header=False)
df_train = pd.DataFrame(Ind_train)
df_train.to_csv(str(model_number) + '_Ind_train.csv', index=False, header=False)
# define model
t0 = time.time()
model = Sequential()
model.add(LSTM(units, stateful=False, input_shape=(n_steps_in, n_features)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(units, stateful=False, return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(n_features)))
model.summary()
model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mse'])
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./tf/logs_" + model_name)
# fit model
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[tf_callback])
print("history.history:{}".format(history.history))
model.save(model_name + '_'
           + str(voronoi_mesh)
           + '_mesh_'
           + str(vor_data_pca_map.shape[1])
           + 'fra_2c_'
           + str(units)
           + 'u_'
           + str(epochs)
           + 'p_'
           + str(batch_size)
           + 'b_r.h5')

t1 = time.time()
train_loss = history.history['loss']
val_loss = history.history['val_loss']
loss_df = pd.DataFrame({'Train Loss': train_loss, 'Validation Loss': val_loss})
loss_df.to_csv(str(model_number) + '_loss.csv', index=False)
print('--------------------finish-------------------')
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


def Cal_T(Ind, num):
    if Ind[num] <= 167:
        T = (Ind[num] + down_lim) * 0.01 + 0.33
        void_shape = 'Ell-x0y0z0'
    return T, void_shape


def relative_err(target, predict):
    return np.linalg.norm(target - predict) / np.linalg.norm(target)


def reverse_data(y, data_len, void_num, n_steps_out):
    reverse_data_all = []
    for j in range(void_num):
        y_data = y[0 + data_len * j:data_len + data_len * j]
        for i in range(data_len):
            if i == 0:
                reverse_data = y_data[i, :n_steps_out - 1].tolist()
            reverse_data.append(y_data[i][n_steps_out - 1])
        reverse_data = np.array(reverse_data)
        reverse_data_all.append(reverse_data)
    return np.array(reverse_data_all)


# recurrently predict
step_train = len(vor_data_pca_map_train)
step = int(len(X_train)/step_train)
# pre——train
yyy_pre_train = []
void_num = int(len(X_train)/step)
for void in range(void_num):
    input = X_train[int(0+void*step)].reshape(1,X_train.shape[1],X_train.shape[2])
    yy_pre_train = input.reshape(X_train.shape[1], X_train.shape[2])
    for j in range(step+(n_steps_out-1)-overlap):
        predict = model.predict(input)
        yy_pre_train = np.vstack((yy_pre_train, predict[0, :1, :]))
        input = yy_pre_train[j+1:j+n_steps_in+1,].reshape(1,X_train.shape[1],X_train.shape[2])
    yyy_pre_train.append(yy_pre_train)
yyy_pre_train = np.array(yyy_pre_train)

# pre——test
yyy_pre_test = []
void_num = int(len(X_test)/step)
for void in range(void_num):
    input = X_test[int(0+void*step)].reshape(1,X_test.shape[1],X_test.shape[2])
    yy_pre_test = input.reshape(X_test.shape[1], X_test.shape[2])
    for j in range(step+(n_steps_out-1)-overlap):
        predict = model.predict(input)
        yy_pre_test = np.vstack((yy_pre_test, predict[0, :1, :]))
        input = yy_pre_test[j+1:j+n_steps_in+1,].reshape(1,X_test.shape[1],X_test.shape[2])
    yyy_pre_test.append(yy_pre_test)
yyy_pre_test = np.array(yyy_pre_test)
# 循Reverse map to original magnitude
vor_data_pca_test_recurrent = np.zeros_like(yyy_pre_test)
vor_data_pca_test_recurrent = np.reshape(vor_data_pca_test_recurrent,(yyy_pre_test.shape[0]*yyy_pre_test.shape[1],-1))

vor_data_pca_train_recurrent = np.zeros_like(yyy_pre_train)
vor_data_pca_train_recurrent = np.reshape(vor_data_pca_train_recurrent,(yyy_pre_train.shape[0]*yyy_pre_train.shape[1],-1))

for component in range(n_component):
    data_flatten = vor_data_pca[:, component:component + 1]
    scaler_data_pca = MinMaxScaler(feature_range=(-1, 1))
    scaler_data_pca.fit(data_flatten)
    vor_data_pca_test_recurrent_map_flatten = np.reshape(yyy_pre_test[:, :, component:component + 1], (-1, 1))
    vor_data_pca_test_recurrent_flatten = scaler_data_pca.inverse_transform(vor_data_pca_test_recurrent_map_flatten)
    vor_data_pca_train_recurrent_map_flatten = np.reshape(yyy_pre_train[:, :, component:component + 1], (-1, 1))
    vor_data_pca_train_recurrent_flatten = scaler_data_pca.inverse_transform(vor_data_pca_train_recurrent_map_flatten)
    vor_data_pca_test_recurrent[:, component:component + 1] = vor_data_pca_test_recurrent_flatten
    vor_data_pca_train_recurrent[:, component:component + 1] = vor_data_pca_train_recurrent_flatten


vor_data_pca_test_recurrent = np.reshape(vor_data_pca_test_recurrent,(yyy_pre_test.shape[0], yyy_pre_test.shape[1], -1))
vor_data_pca_train_recurrent = np.reshape(vor_data_pca_train_recurrent,(yyy_pre_train.shape[0], yyy_pre_train.shape[1], -1))
# -------------------------------------------------

y_pre_train = model.predict(X_train)
y_pre_test = model.predict(X_test)
err_train = relative_err(y_train, y_pre_train)
err_test = relative_err(y_test, y_pre_test)
print('train err is: ', err_train)
print('test err is: ', err_test)

save_number_to_txt(err_test, 'relative error-test.txt')
step_train = len(vor_data_pca_map_train)
step = int(len(X_train) / step_train)
y_train_pre_map = reverse_data(y_pre_train, step, int(len(y_pre_train) / step), n_steps_out)
y_test_pre_map = reverse_data(y_pre_test, step, int(len(y_pre_test) / step), n_steps_out)

vor_data_pca_test_ori = np.zeros_like(vor_data_pca_map_test)
vor_data_pca_test_ori = np.reshape(vor_data_pca_test_ori,
                                   (vor_data_pca_map_test.shape[0] * vor_data_pca_map_test.shape[1], -1))
vor_data_pca_train_ori = np.zeros_like(vor_data_pca_map_train)
vor_data_pca_train_ori = np.reshape(vor_data_pca_train_ori,
                                    (vor_data_pca_map_train.shape[0] * vor_data_pca_map_train.shape[1], -1))
vor_data_pca_test_pre_one = np.zeros_like(y_test_pre_map)
vor_data_pca_test_pre_one = np.reshape(vor_data_pca_test_pre_one,
                                       (y_test_pre_map.shape[0] * y_test_pre_map.shape[1], -1))
vor_data_pca_train_pre_one = np.zeros_like(y_train_pre_map)
vor_data_pca_train_pre_one = np.reshape(vor_data_pca_train_pre_one,
                                        (y_train_pre_map.shape[0] * y_train_pre_map.shape[1], -1))
for component in range(n_component):

    data_flatten = vor_data_pca[:, component:component + 1]
    scaler_data_pca = MinMaxScaler(feature_range=(-1, 1))
    scaler_data_pca.fit(data_flatten)

    vor_data_pca_map_test_flatten = np.reshape(vor_data_pca_map_test[:, :, component:component + 1], (-1, 1))
    vor_data_pca_test_ori_flatten = scaler_data_pca.inverse_transform(vor_data_pca_map_test_flatten)

    vor_data_pca_map_train_flatten = np.reshape(vor_data_pca_map_train[:, :, component:component + 1], (-1, 1))
    vor_data_pca_train_ori_flatten = scaler_data_pca.inverse_transform(vor_data_pca_map_train_flatten)

    vor_data_pca_test_ori[:, component:component + 1] = vor_data_pca_test_ori_flatten
    vor_data_pca_train_ori[:, component:component + 1] = vor_data_pca_train_ori_flatten

    vor_data_pca_map_test_pre_one_flatten = np.reshape(y_test_pre_map[:, :, component:component + 1], (-1, 1))
    vor_data_pca_test_pre_one_flatten = scaler_data_pca.inverse_transform(vor_data_pca_map_test_pre_one_flatten)

    vor_data_pca_map_train_pre_one_flatten = np.reshape(y_train_pre_map[:, :, component:component + 1], (-1, 1))
    vor_data_pca_train_pre_one_flatten = scaler_data_pca.inverse_transform(vor_data_pca_map_train_pre_one_flatten)

    vor_data_pca_test_pre_one[:, component:component + 1] = vor_data_pca_test_pre_one_flatten
    vor_data_pca_train_pre_one[:, component:component + 1] = vor_data_pca_train_pre_one_flatten


vor_data_pca_test_ori = np.reshape(vor_data_pca_test_ori,
                                   (vor_data_pca_map_test.shape[0], vor_data_pca_map_test.shape[1], -1))
vor_data_pca_train_ori = np.reshape(vor_data_pca_train_ori,
                                    (vor_data_pca_map_train.shape[0], vor_data_pca_map_train.shape[1], -1))
vor_data_pca_test_pre_one = np.reshape(vor_data_pca_test_pre_one,
                                       (y_test_pre_map.shape[0], y_test_pre_map.shape[1], -1))
vor_data_pca_train_pre_one = np.reshape(vor_data_pca_train_pre_one,
                                        (y_train_pre_map.shape[0], y_train_pre_map.shape[1], -1))


# pca
vor_train_recurrent_inverse = pca.inverse_transform(vor_data_pca_train_recurrent)
vor_test_recurrent_inverse = pca.inverse_transform(vor_data_pca_test_recurrent)

vor_train_pre_one_inverse = pca.inverse_transform(vor_data_pca_train_pre_one)
vor_test_pre_one_inverse = pca.inverse_transform(vor_data_pca_test_pre_one)
vor_train_ori_inverse = pca.inverse_transform(vor_data_pca_train_ori)
vor_test_ori_inverse = pca.inverse_transform(vor_data_pca_test_ori)

vor_train_recurrent_noisy = np.reshape(vor_train_recurrent_inverse, (vor_train_recurrent_inverse.shape[0], vor_train_recurrent_inverse.shape[1], voronoi_mesh, voronoi_mesh))
vor_test_recurrent_noisy = np.reshape(vor_test_recurrent_inverse, (vor_test_recurrent_inverse.shape[0], vor_test_recurrent_inverse.shape[1], voronoi_mesh, voronoi_mesh))

vor_train_pre_one_noisy = np.reshape(vor_train_pre_one_inverse, (vor_train_pre_one_inverse.shape[0], vor_train_pre_one_inverse.shape[1], voronoi_mesh, voronoi_mesh))
vor_test_pre_one_noisy = np.reshape(vor_test_pre_one_inverse,(vor_test_pre_one_inverse.shape[0], vor_test_pre_one_inverse.shape[1], voronoi_mesh, voronoi_mesh))

vor_train_ori_noisy = np.reshape(vor_train_ori_inverse,(vor_train_ori_inverse.shape[0], vor_train_ori_inverse.shape[1], voronoi_mesh, voronoi_mesh))
vor_test_ori_noisy = np.reshape(vor_test_ori_inverse,(vor_test_ori_inverse.shape[0], vor_test_ori_inverse.shape[1], voronoi_mesh, voronoi_mesh))
print('pca inverse finish')

np.save('./' + str(model_number) + '_vor_train_recurrent_noisy.npy', vor_train_recurrent_noisy)
np.save('./' + str(model_number) + '_vor_test_recurrent_noisy.npy', vor_test_recurrent_noisy)

np.save('./' + str(model_number) + '_vor_train_pre_one_noisy.npy', vor_train_pre_one_noisy)
np.save('./' + str(model_number) + '_vor_test_pre_one_noisy.npy', vor_test_pre_one_noisy)

np.save('./' + str(model_number) + '_vor_train_ori_noisy.npy', vor_train_ori_noisy)
np.save('./' + str(model_number) + '_vor_test_ori_noisy.npy', vor_test_ori_noisy)

vor_test_recurrent = vor_test_recurrent_noisy
vor_train_recurrent = vor_train_recurrent_noisy

vor_test_pre_one = vor_test_pre_one_noisy
vor_train_pre_one = vor_train_pre_one_noisy

vor_train_ori = vor_train_ori_noisy
vor_test_ori = vor_test_ori_noisy


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


void_data = np.delete(void_data_all_cal, delete, axis=3)
void_data = void_data.reshape(-1, void_data.shape[2], void_data.shape[3])
Xmax_void, Xmin_void = void_data[:, :, 0].max(), void_data[:, :, 0].min()
Ymax_void, Ymin_void = void_data[:, :, 1].max(), void_data[:, :, 1].min()

sca_ori = inverse_voronoi(vor_data,Xmin_void, Xmax_void, Ymin_void, Ymax_void)

sca_test_pre_recurrent = inverse_voronoi(vor_test_recurrent ,Xmin_void, Xmax_void, Ymin_void, Ymax_void)
sca_train_pre_recurrent = inverse_voronoi(vor_train_recurrent ,Xmin_void, Xmax_void, Ymin_void, Ymax_void)

sca_test_pre = inverse_voronoi(vor_test_pre_one ,Xmin_void, Xmax_void, Ymin_void, Ymax_void)
sca_train_pre = inverse_voronoi(vor_train_pre_one,Xmin_void, Xmax_void, Ymin_void, Ymax_void)

sca_test_ori_pca = inverse_voronoi(vor_test_ori ,Xmin_void, Xmax_void, Ymin_void, Ymax_void)
sca_train_ori_pca = inverse_voronoi(vor_train_ori,Xmin_void, Xmax_void, Ymin_void, Ymax_void)
print('sca finish')
np.save('./' + str(model_number) + '_sca_ori.npy', sca_ori)
np.save('./' + str(model_number) + '_sca_test_pre_recurrent.npy',sca_test_pre_recurrent)
np.save('./' + str(model_number) + '_sca_train_pre_recurrent.npy',sca_train_pre_recurrent)
np.save('./' + str(model_number) + '_sca_test_pre.npy', sca_test_pre)
np.save('./' + str(model_number) + '_sca_train_pre.npy', sca_train_pre)
np.save('./' + str(model_number) + '_sca_test_ori_pca.npy', sca_test_ori_pca)
np.save('./' + str(model_number) + '_sca_train_ori_pca.npy', sca_train_ori_pca)


# S/S0
data_S0 = sca_ori[:1,0:1,:,:].reshape(-1, 3)
data_cleaned = data_S0[~np.isnan(data_S0).any(axis=1)]
data_cleaned_xy = data_cleaned[:,:2]
hull = ConvexHull(data_cleaned_xy)
S0 = hull.volume


def Ell_S_data(void_data, S0):
    void_num = void_data.shape[0]
    void_frame = void_data.shape[1]
    Ell_S = np.zeros((void_num,void_frame,3))
    Ell_S[:, :, 0] = S0
    for num in range(void_num):
        for frame in range(void_frame):
            data = void_data[num:num+1, frame:frame+1, :, :].reshape(-1, 3)
            data_cleaned = data[~np.isnan(data).any(axis=1)]
            data_cleaned_xy = data_cleaned[:, :2]
            hull = ConvexHull(data_cleaned_xy)
            S = hull.volume
            Ell_S[num][frame][1] = S
            Ell_S[num][frame][2] = S/S0
    return Ell_S


Ell_S_ori = Ell_S_data(sca_ori,S0)
Ell_S_test_ori = Ell_S_ori[Ind_test]
Ell_S_train_ori = Ell_S_ori[Ind_train]

Ell_S_test_ori_pca = Ell_S_data(sca_test_ori_pca,S0)
Ell_S_train_ori_pca = Ell_S_data(sca_train_ori_pca,S0)

Ell_S_test_pre_one = Ell_S_data(sca_test_pre,S0)
Ell_S_train_pre_one = Ell_S_data(sca_train_pre,S0)

Ell_S_test_pre_recurrent = Ell_S_data(sca_test_pre_recurrent,S0)
Ell_S_train_pre_recurrent = Ell_S_data(sca_train_pre_recurrent,S0)


np.save('./' + str(model_number) + '_Ell_S_test_ori.npy', Ell_S_test_ori)
np.save('./' + str(model_number) + '_Ell_S_train_ori.npy', Ell_S_train_ori)

np.save('./' + str(model_number) + '_Ell_S_test_ori_pca.npy', Ell_S_test_ori_pca)
np.save('./' + str(model_number) + '_Ell_S_train_ori_pca.npy', Ell_S_train_ori_pca)

np.save('./' + str(model_number) + '_Ell_S_test_pre_one.npy', Ell_S_test_pre_one)
np.save('./' + str(model_number) + '_Ell_S_train_pre_one.npy', Ell_S_train_pre_one)

np.save('./' + str(model_number) + '_Ell_S_test_pre_recurrent.npy', Ell_S_test_pre_recurrent)
np.save('./' + str(model_number) + '_Ell_S_train_pre_recurrent.npy', Ell_S_train_pre_recurrent)


# point-wise error
def point_wise_error(true_curve1, fit_curve2):
    relative_error = sum(abs(y1 - y2) / y1 if y1 != 0 else 0 for y1, y2 in zip(true_curve1, fit_curve2))
    return relative_error


error_test = 0
for i in range(len(Ell_S_test_ori_pca)):
    error_test = error_test + point_wise_error(Ell_S_test_ori_pca[i, n_steps_out:, 2], Ell_S_test_pre_one[i, 0:, 2])
error_test = error_test/len(Ell_S_test_ori_pca)

save_number_to_txt(error_test, 'point-wise error_test.txt')
print('point-wise error_test=',error_test)




