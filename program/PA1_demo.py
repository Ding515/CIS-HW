# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 10:40:19 2021

@author: Ding
"""
import numpy as np
import PA1_functions as PA1f
folder_director='C:\\Users\\Ding\\Desktop\\JHU第一学期作业\\CIS\\Programming_hw1\\2021_pa_1-2_student_data\\2021 PA 1-2 Student Data\\pa1-debug-'
file_name='a'
calbody_name='-calbody.txt'
calreading_name='-calreadings.txt'
calbody_files=folder_director+file_name+calbody_name
calreading_files=folder_director+file_name+calreading_name

print('current case:',file_name)
d=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[0:8,:]
a=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[8:16,:]
c=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[16:43,:]
D=np.loadtxt(calreading_files,skiprows=1,delimiter=',')[0:8,:]
A=np.loadtxt(calreading_files,skiprows=1,delimiter=',')[8:16,:]
C=np.loadtxt(calreading_files,skiprows=1,delimiter=',')[16:43,:]

#test of rotation part
vector =np.array([1,2,3])
R=np.array([[0.36,0.48,-0.8],[-0.8,0.6,0],[0.48,0.64,0.6]])
vector_T=PA1f.rotation(R,vector)

F_D_qua=PA1f.points_registeration_q1(d,D)
print('Quaternion error of D',((PA1f.frame_transformation(F_D_qua[:,0:3],F_D_qua[:,3].reshape(3,1),np.transpose(d))-np.transpose(D))**2).mean(axis=None))
F_A_qua=PA1f.points_registeration_q1(a,A)
print('Quaternion error of A',((PA1f.frame_transformation(F_A_qua[:,0:3],F_A_qua[:,3].reshape(3,1),np.transpose(a))-np.transpose(A))**2).mean(axis=None))
F_D_SVD=PA1f.points_registeration(d,D)
print('SVD error of D',((PA1f.frame_transformation(F_D_SVD[:,0:3],F_D_SVD[:,3].reshape(3,1),np.transpose(d))-np.transpose(D))**2).mean(axis=None))
F_A_SVD=PA1f.points_registeration_q1(a,A)
print('SVD error of A',((PA1f.frame_transformation(F_A_SVD[:,0:3],F_A_SVD[:,3].reshape(3,1),np.transpose(a))-np.transpose(A))**2).mean(axis=None))

C_expect=PA1f.distortion_calibration_direct(d,D,a,A,np.transpose(c))

print('C error',((np.transpose(C)-C_expect)**2).mean(axis=None))

#folder_director='C:\\Users\\Ding\\Desktop\\JHU第一学期作业\\CIS\\Programming_hw1\\2021_pa_1-2_student_data\\2021 PA 1-2 Student Data\\pa1-unknown-'
#save_director='C:\\Users\\Ding\\Desktop\\JHU第一学期作业\\CIS\\Programming_hw1\\PROGRAMS\\output_of_C\\'
#for file_name in ['h','i','j','k']:
#    calbody_files=folder_director+file_name+calbody_name
#    calreading_files=folder_director+file_name+calreading_name
#    d=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[0:8,:]
#    a=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[8:16,:]
#    c=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[16:43,:]
#    D=np.loadtxt(calreading_files,skiprows=1,delimiter=',')[0:8,:]
#    A=np.loadtxt(calreading_files,skiprows=1,delimiter=',')[8:16,:]
#    C_expect=PA1f.distortion_calibration_direct(d,D,a,A,np.transpose(c))
#    save_path=save_director+'C_expect_case_'+file_name+'.npy'
#    np.save(save_path,np.transpose(C_expect))