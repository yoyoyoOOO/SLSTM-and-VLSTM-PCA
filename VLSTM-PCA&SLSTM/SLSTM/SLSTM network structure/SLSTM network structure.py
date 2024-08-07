"""
SLSTM network structure
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from numpy import array
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras.optimizers import RMSprop, Adam
from random import shuffle
import pandas as pd
import tensorflow as tf
import math

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# # Setting the work directory
dir_parameters = 'x0y0z0_100fra_CH_cubic_w20'
dir1 = 'D:\\pycharm_file\\3.AI-AL\\'
dir2 = 'D:\\pycharm_file\\3.AI-AL\\SLSTM network structure'
# LSTM
n_steps_in, n_steps_out, n_features, overlap = 5, 5, 3, 0
ratio_split = 0.8
rms = RMSprop(learning_rate=0.001, decay=0.0001)
adam = Adam(learning_rate=0.001, decay=0.0001)
batch_size = 16
epochs = 200
units = 100
down_lim = 0
# Name
model_number = 1
name_index = dir_parameters.find("_")
w = dir_parameters[-3:]
model_name = str(model_number)+'_'+dir_parameters[:name_index]+'_'+str(168-down_lim)+'void_'+w
# Ell_data 
spherical_r = 0.00288
Profile_len = 0.079

# ------------------------------------------------------------------------------
# load data
os.chdir(dir1+dir_parameters)
eq_polar_axis_all = np.load(file='./eq_polar_axis_all.npy')
eqstrain_division_points = np.load(file='./eqstrain_division_points_AL.npy')
eqstrain_division_points_expanded = np.expand_dims(eqstrain_division_points, axis=-1)
polar_eq_all = np.concatenate((eq_polar_axis_all, eqstrain_division_points_expanded), axis=-1)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder create  ---")
    else:
        print("---  There is this folder!  ---")


mkdir(dir2)
os.chdir(dir2)

#  V/V0
def Ell_data(ori_data, spherical_r, Profile_len):
    Volume_Profile = Profile_len ** 3
    Volume_spherical = math.pi * 4 / 3 * spherical_r ** 3
    f0 = Volume_spherical / Volume_Profile
    data = np.zeros((ori_data.shape[0], ori_data.shape[1], ori_data.shape[2]+4))
    data[:, :, :ori_data.shape[2]] = ori_data
    # V = (4/3) * π * a * b * c
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            a = data[i, j, 0]
            b = data[i, j, 1]
            c = data[i, j, 2]
            Volume_spherical_pre = (4/3) * np.pi * a * b * c
            data[i, j, ori_data.shape[2] - 1 + 1] = Volume_Profile
            data[i, j, ori_data.shape[2] - 1 + 2] = Volume_spherical
            data[i, j, ori_data.shape[2] - 1 + 3] = Volume_spherical_pre
            f = Volume_spherical_pre / Volume_Profile
            data[i, j, ori_data.shape[2] - 1 + 4] = f/f0
    return data


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
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-overlap:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# Scale data--------------------------------------
# axis a
SMA_a = polar_eq_all[down_lim:,:,0:1]
SMA_a_flatten = np.reshape(SMA_a, [SMA_a.shape[0]*SMA_a.shape[1], 1])
scaler_SMA_a = MinMaxScaler(feature_range=(0, 1))
scaler_SMA_a.fit(SMA_a_flatten)
SMA_a_flatten_map = scaler_SMA_a.transform(SMA_a_flatten)
SMA_a_map = np.reshape(SMA_a_flatten_map, [SMA_a.shape[0], SMA_a.shape[1], 1])
# axis b
SMA_b = polar_eq_all[down_lim:,:,1:2]
SMA_b_flatten = np.reshape(SMA_b, [SMA_b.shape[0]*SMA_b.shape[1], 1])
scaler_SMA_b = MinMaxScaler(feature_range=(0, 1))
scaler_SMA_b.fit(SMA_b_flatten)
SMA_b_flatten_map = scaler_SMA_b.transform(SMA_b_flatten)
SMA_b_map = np.reshape(SMA_b_flatten_map, [SMA_b.shape[0], SMA_b.shape[1], 1])
# axis c
SMA_c = polar_eq_all[down_lim:,:,2:3]
SMA_c_flatten = np.reshape(SMA_c, [SMA_c.shape[0]*SMA_c.shape[1], 1])
scaler_SMA_c = MinMaxScaler(feature_range=(0, 1))
scaler_SMA_c.fit(SMA_c_flatten)
SMA_c_flatten_map = scaler_SMA_c.transform(SMA_c_flatten)
SMA_c_map = np.reshape(SMA_c_flatten_map, [SMA_c.shape[0], SMA_c.shape[1], 1])
# Scale data--------------------------------------finish
SMA_map = np.concatenate((SMA_a_map,SMA_b_map,SMA_c_map), axis=-1)

# into lstm
Ind = list(range(len(SMA_map)))
shuffle(Ind)   # Random disruption of data
Ind_train = Ind[0:round(ratio_split * len(SMA_map))]
Ind_test = Ind[round(ratio_split * len(SMA_map)):]
SMA_train = SMA_map[Ind_train]
SMA_test = SMA_map[Ind_test]
# training dataset
X_train = np.empty((0, n_steps_in, n_features))
y_train = np.empty((0, n_steps_in, n_features))
for i in range(len(SMA_train)):
    X_coord_eq, y_coord_eq = split_sequences(SMA_train[i], n_steps_in, n_steps_out, overlap)
    X_train = np.concatenate((X_train, X_coord_eq), axis=0)
    y_train = np.concatenate((y_train, y_coord_eq), axis=0)
# testing dataset
X_test = np.empty((0, n_steps_in, n_features))
y_test = np.empty((0, n_steps_in, n_features))
for i in range(len(SMA_test)):
    X_coord_eq, y_coord_eq = split_sequences(SMA_test[i], n_steps_in, n_steps_out, overlap)
    X_test = np.concatenate((X_test, X_coord_eq), axis=0)
    y_test = np.concatenate((y_test, y_coord_eq), axis=0)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

df_test = pd.DataFrame(Ind_test)
df_test .to_csv(str(model_number)+'_Ind_test.csv', index=False, header=False)
df_train = pd.DataFrame(Ind_train)
df_train .to_csv(str(model_number)+'_Ind_train.csv', index=False, header=False)
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
model.compile(optimizer=adam, loss='mean_squared_error',metrics=['mse'])
tf_callback = tf.keras.callbacks.TensorBoard(log_dir="./tf/logs_"+model_name)
# fit model
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    callbacks=[tf_callback])
print("history.history:{}".format(history.history))
model.save(model_name+'_'
           +str(SMA_map.shape[1])
           +'fra_2c_'
           +str(units)
           +'u_'
           +str(epochs)
           +'p_'
           +str(batch_size)
           +'b_r_loop.h5')

t1 = time.time()
# loss data
train_loss = history.history['loss']
val_loss = history.history['val_loss']
loss_df = pd.DataFrame({'Train Loss': train_loss, 'Validation Loss': val_loss})
loss_df.to_csv(str(model_number)+'_loss.csv', index=False)
print('--------------------finish-------------------')
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


def Cal_T(Ind, num):
    if Ind[num]<=167:
        T = (Ind[num]+down_lim)*0.01+0.33
        void_shape = 'Ell-x0y0z0'
    return T, void_shape


def relative_err(target, predict):
    return np.linalg.norm(target - predict)/np.linalg.norm(target)


# Reverse map to original magnitude
def Reverse_map(data, scaler_SMA_a, scaler_SMA_b, scaler_SMA_c):
    # SMA-a
    SMA_a_data = data[:, :, :1].reshape(data.shape[0] * data.shape[1], 1)
    SMA_a_data_rev = scaler_SMA_a.inverse_transform(SMA_a_data)
    SMA_a_data_rev = SMA_a_data_rev.reshape(data.shape[0], data.shape[1], 1)
    # SMA-b
    SMA_b_data = data[:, :, 1:2].reshape(data.shape[0] * data.shape[1], 1)
    SMA_b_data_rev = scaler_SMA_b.inverse_transform(SMA_b_data)
    SMA_b_data_rev = SMA_b_data_rev.reshape(data.shape[0], data.shape[1], 1)
    # SMA-b
    SMA_c_data = data[:, :, 2:3].reshape(data.shape[0] * data.shape[1], 1)
    SMA_c_data_rev = scaler_SMA_c.inverse_transform(SMA_c_data)
    SMA_c_data_rev = SMA_c_data_rev.reshape(data.shape[0], data.shape[1], 1)
    SMA_data_pre = np.concatenate((SMA_a_data_rev, SMA_b_data_rev, SMA_c_data_rev), axis=-1)
    return SMA_data_pre


def reverse_data(y, data_len, void_num, n_steps_out):
    reverse_data_all = []
    for j in range(void_num):
        y_data = y[0+data_len*j:data_len+data_len*j]
        for i in range(data_len):
            if i == 0:
                reverse_data = y_data[i,:n_steps_out-1].tolist()
            reverse_data.append(y_data[i][n_steps_out-1])
        reverse_data = np.array(reverse_data)
        reverse_data_all.append(reverse_data)
    return np.array(reverse_data_all)


y_pre_train = model.predict(X_train)
y_pre_test = model.predict(X_test)
err_train = relative_err(y_train, y_pre_train)
err_test = relative_err(y_test, y_pre_test)
print('test err is: ', err_test)

step_train = len(SMA_train)
step = int(len(X_train)/step_train)
y_train_pre_map = reverse_data(y_pre_train, step, int(len(y_pre_train)/step), n_steps_out)
y_test_pre_map = reverse_data(y_pre_test, step, int(len(y_pre_test)/step), n_steps_out)

SMA_train_pre_one = Reverse_map(y_train_pre_map, scaler_SMA_a, scaler_SMA_b, scaler_SMA_c)
SMA_test_pre_one = Reverse_map(y_test_pre_map, scaler_SMA_a, scaler_SMA_b, scaler_SMA_c)
# One step data--------------------------------------------
# recurrently predict
step_train = len(SMA_train)
step = int(len(X_train)/step_train)
# pre——train
yyy_pre_train = []
void_num = int(len(X_train)/step)
for void in range(void_num):
    input = X_train[int(0+void*step)].reshape(1,X_train.shape[1],X_train.shape[2])
    yy_pre_train = input.reshape(X_train.shape[1], X_train.shape[2])
    for j in range(step+(n_steps_out-1)-overlap):
        print(j)
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

# Reverse map to original magnitude
SMA_train_pre_loop = Reverse_map(yyy_pre_train, scaler_SMA_a, scaler_SMA_b, scaler_SMA_c)
SMA_test_pre_loop = Reverse_map(yyy_pre_test, scaler_SMA_a, scaler_SMA_b, scaler_SMA_c)

# V/V0
Ell_test_pre_loop = Ell_data(SMA_test_pre_loop, spherical_r, Profile_len)
Ell_train_pre_loop = Ell_data(SMA_train_pre_loop, spherical_r, Profile_len)
Ell_test_pre = Ell_data(SMA_test_pre_one, spherical_r, Profile_len)
Ell_train_pre = Ell_data(SMA_train_pre_one, spherical_r, Profile_len)
Tra_data = polar_eq_all[down_lim:,:,:]
Ell_test_ori = Ell_data(Tra_data[Ind_test], spherical_r, Profile_len)
Ell_train_ori = Ell_data(Tra_data[Ind_train], spherical_r, Profile_len)
# npy
np.save('./'+str(model_number)+'_Ell_test_pre_loop.npy', Ell_test_pre_loop)
np.save('./'+str(model_number)+'_Ell_train_pre_loop.npy', Ell_train_pre_loop)
np.save('./'+str(model_number)+'_Ell_test_pre.npy', Ell_test_pre)
np.save('./'+str(model_number)+'_Ell_train_pre.npy', Ell_train_pre)
np.save('./'+str(model_number)+'_Ell_test_ori.npy', Ell_test_ori)
np.save('./'+str(model_number)+'_Ell_train_ori.npy', Ell_train_ori)
