import os
import time
import numpy as np
import pandas as pd
import csv


def nums(first_number, last_number, step=1):
    return range(first_number, last_number + 1, step)


def integrate_to_corer(num_frames, work_dir, inp, originstr, endstr):
    magnitude = len(str(num_frames))
    os.chdir(work_dir)
    t0 = time.time()
    inpfile = open(inp)
    lines = inpfile.readlines()
    inpfile.close()
    ori_index = lines.index(originstr)  # c3d10 starting line
    end_index = lines.index(endstr)  # c3d10 ending line
    newfile = []
    for line in range(ori_index + 1, end_index):
        newfile.append(lines[line].rstrip('\n'))
    newfile = pd.DataFrame(newfile)
    data_c3d10 = newfile[0].str.split(',', expand=True)

    c3d10_void = data_c3d10.iloc[:, :5].astype(int)
    file = pd.read_excel(r'./Code_file/element_number.xls')
    element = file['elementLabel'].values
    element = list(set(element))
    ele_num = np.empty((0, 5)).astype(int)
    
    for elem in element:
        void_num = c3d10_void[c3d10_void[0] == elem].iloc[:, :].values
        ele_num = np.concatenate((ele_num, void_num))
        
    # sorted
    ele_num = ele_num[np.argsort(ele_num[:, 0])]
    np.savetxt(r'ele_four.csv', ele_num, fmt='%d', delimiter=',')
    
    COORD = pd.read_excel(r'./Code_file/node_number.xls')
    COORD_num = COORD['nodeLabel'].values
    # delete
    data = np.delete(ele_num, 0, axis=1)
    void_vertex = np.empty((0, 3)).astype(int)
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            if data[i][j] not in COORD_num:
                temp = np.delete(data[i], j)
                void_vertex = np.vstack((void_vertex, temp))
                break
    np.savetxt(r'vertex.csv', void_vertex, fmt='%d', delimiter=',')
    vertex_num = np.sort(list(set(void_vertex.flatten())))
    Coord_vertex = COORD[COORD['nodeLabel'] == vertex_num[0]]
    np.savetxt(r'vertex_num.csv'
               , vertex_num, fmt='%d', delimiter=',')

    # int_coord
    int_coord = [[0.5854102, 0.1381966, 0.1381966, 1], [0.1381966, 0.5854102, 0.1381966, 1],
                 [0.1381966, 0.1381966, 0.5854102, 1],
                 [0.1381966, 0.1381966, 0.1381966, 1]]
    int_coord = np.matrix(int_coord)
    # 1-4label node
    point_4 = [0, 0, 0, 1]
    point_1 = [1, 0, 0, 1]
    point_2 = [0, 1, 0, 1]
    point_3 = [0, 0, 1, 1]
    point_5 = [0.5, 0, 0, 1]
    point_6 = [0, 0.5, 0, 1]
    point_7 = [0, 0, 0.5, 1]
    point_8 = [0.5, 0.5, 0, 1]
    point_9 = [0, 0.5, 0.5, 1]
    point_10 = [0.5, 0, 0.5, 1]
    point = np.vstack((point_1, point_2, point_3, point_4))

    # element number
    ele_four = pd.read_csv(r'ele_four.csv', header=None)
    ele_four = ele_four.to_numpy()
    ele_four = np.delete(ele_four, 0, axis=1)

    node = pd.read_csv(r'vertex_num.csv', header=None)
    for snn in range(num_frames):
        # int node
        data_all = pd.read_excel('./integration/snn_integration' + str(snn).zfill(magnitude) + '.xls', sheet_name='sheet1')
        data_all = data_all.drop(['integrationPoint'], axis=1)
        data = data_all.iloc[:, :]
        data = np.array(data)
        # Si
        A = []
        ele_num = []
        int_coord_1 = np.linalg.inv(int_coord)
        for i in nums(1, int(len(data) / 4)):
            ele_num1 = data[4 * (i - 1)][0]
            ele_num.append(ele_num1)
            cal = data[4 * (i - 1):4 * i, 1:]
            coef = int_coord_1 * cal
            A.append(coef)
        # node stress
        snn_cal = []
        for i in range(len(A)):
            snn_cal1 = point * A[i]
            snn_cal.append(snn_cal1)
        res_all = []
        for i in range(len(node)):
            sum_cal = []
            index = np.where(ele_four == node[0][i])
            lencol = np.asarray(index)
            for j in range(lencol.shape[1]):
                row = index[0][j]
                col = index[1][j]
                data_cal = np.array(snn_cal[row][col]).flatten()
                sum_cal.append(data_cal)
            res = np.mean(sum_cal, 0)
            res = res.tolist()
            res.insert(0, node[0][i])
            res_all.append(res)
        snn_date = np.delete(res_all, 0, axis=1)
        mat = np.zeros((3, 3))
        three_S = []
        for i in range(len(snn_date)):
            mat[0, 0] = snn_date[i][0]
            mat[1, 1] = snn_date[i][1]
            mat[2, 2] = snn_date[i][2]
            mat[0, 1] = snn_date[i][3]
            mat[1, 0] = snn_date[i][3]
            mat[0, 2] = snn_date[i][4]
            mat[2, 0] = snn_date[i][4]
            mat[1, 2] = snn_date[i][5]
            mat[2, 1] = snn_date[i][5]
            eigenvalue, featurevector = np.linalg.eig(mat)
            eigenvalue = sorted(eigenvalue, reverse=True)
            eigenvalue.insert(0, node[0][i])
            three_S.append(eigenvalue)
        #  Mises = sqrt(0.5*((s11-s22)^2 + (s22-s33)^2 + (s33-s11)^2 + 6*(s12^2 + s23^2 + s13^2)))
        snn_date = np.delete(res_all, 0, axis=1)
        Mises = []
        for i in range(len(snn_date)):
            Mis = []
            Mi = (((snn_date[i][0] - snn_date[i][1]) ** 2 + (snn_date[i][0] - snn_date[i][2]) ** 2 + (
                        snn_date[i][1] - snn_date[i][2]) ** 2
                   + 6 * (snn_date[i][3] ** 2 + snn_date[i][4] ** 2 + snn_date[i][5] ** 2)) / 2) ** 0.5
            Mis.append(Mi)
            Mis.insert(0, node[0][i])
            Mises.append(Mis)
        # res_all=s11 s22 s33 s12 s13 s23
        with open('./corner_node/snn_cornerpoint_' + str(snn).zfill(magnitude) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in res_all:
                writer.writerow(row)
        # three_S
        with open('./corner_node/three_S_cornerpoint_' + str(snn).zfill(magnitude) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in three_S:
                writer.writerow(row)
        # Mises
        with open('./corner_node/Mises_cornerpoint_' + str(snn).zfill(magnitude) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in Mises:
                writer.writerow(row)
        # Remove the center point
        coor_date = './corner_node/COORD_corner' + str(snn).zfill(magnitude) + '.csv'
        codata = pd.read_csv(coor_date)
        row_to_delete = len(vertex_num)
        codata = codata[:row_to_delete]
        codata.to_csv(coor_date, index=False)
    "finish integration to corner"


num_frames = 101
originstr = "*Element, type=C3D10\n"           
endstr = "*Nset, nset=WHOLE, generate\n"     
trixx = 0.33
T1 = []
T2 = []
void_num = 168
for i in range(void_num):
    TName2 = 'T' + str(int(trixx*100)).zfill(len(str(void_num)))
    TName1 = 'T' + str(round(trixx, 3)).replace('.', '_')
    T1.append(TName1)
    T2.append(TName2)
    trixx = trixx+0.01


x_angle = ['0']
y_angle = ['0']
z_angle = ['0']
shape = ['ell', 'sph']
Axial_ratio = '02'
void_shape = shape[0]
base_dir = 'D:\\ABAQUS_wxj\\RVE_6061AL\\all data\\ell\\'
for j in range(len(x_angle)):
    X_Rotateangle = x_angle[j]
    for n in range(len(y_angle)):
        Y_Rotateangle = y_angle[n]
        for m in range(len(z_angle)):
            Z_Rotateangle = z_angle[m]
            for i in range(len(T1)):
                void_name1 = void_shape + Axial_ratio + T1[i]
                void_name2 = void_shape + Axial_ratio + T2[i]
                print(void_name2+' begin')
                inp = void_name1 + '.inp'

                work_dir = base_dir+'x'+X_Rotateangle + 'y'+Y_Rotateangle+'z'+Z_Rotateangle+'_168void_whole_eq_AL_w05\\void-'+void_name2

                integrate_to_corer(num_frames, work_dir, inp, originstr, endstr)
                print('x'+X_Rotateangle + 'y'+Y_Rotateangle+'z'+Z_Rotateangle + void_name2+'finish')

