# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 05:45:25 2021

@author: Ding
"""
import numpy as np
from scipy.special import comb
import PA1_functions as PAf1
################### QUESTION 1 FUNCTION PART ########################
"""
Input: 
    folder_director:where to load the 2021 PA 1-2 Student Data
    
    save_file_path: where to save the output C_{expect} data, please note that considering the 
    size of each data set we do not return them for the following problems but load them again
    from the SAVE_FILE_PATH
    
    function_status: Two modes for datasets debug and unknown
    
    file_dictionary: lists of dataset to be processed
    
OUTPUT:
    No direct returns, C_expect is saved in the save_file_path by '_C_expect.npy'
"""
def C_expect_output(folder_director,save_file_path,function_status,dictionary):
    
    ##########GENERAL SETTING AND DATA LOADING##################
    calbody_name='-calbody.txt'
    calreading_name='-calreadings.txt'
    for file_name in dictionary:
        calbody_files=folder_director+function_status+'-'+file_name+calbody_name
        calreading_files=folder_director+function_status+'-'+file_name+calreading_name
        
        print('current case:',file_name)
        
        f=open(calreading_files)
        line=f.readline()
        n_frame=int(line[10:13])
        n_point=int(line[0]) +int(line[3]) +int(line[6:8])
        d=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[0:8,:]
        a=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[8:16,:]
        c=np.loadtxt(calbody_files,skiprows=1,delimiter=',')[16:43,:]
        C_mat=[]
        for frame in range(n_frame):

            D=np.loadtxt(calreading_files,skiprows=1,delimiter=',')[(0+n_point*frame):(8+n_point*frame),:]
            A=np.loadtxt(calreading_files,skiprows=1,delimiter=',')[(8+n_point*frame):(16+n_point*frame),:]
            
##############COMPUTE THE CORRESPONDING C_EXPECT BASED ON PA1 Q3############################   
            
            C_expect=PAf1.distortion_calibration_direct(d,D,a,A,np.transpose(c))
            C_mat.append(C_expect)
        C_prediction=np.array(C_mat)
        
##################### SAVING PART #############################        
        save_path=save_file_path+file_name+'_C_expect.npy'
        np.save(save_path,C_prediction)
        
        
################### QUESTION 2 FUNCTION PART ########################
"""
scale_to_box is used to normalized the original data between [0,1]

Input: 
    p: Nx3 matrix corresponding to (x,y,z) coordinates, which is normalized one by one.
    
OUTPUT:
    p:The normalized coordination matrix
    
    coordinates_range:The corresponding min max range for each coordinates to save for G normalization
    
"""        
def scale_to_box(p):
    coordinates_range=np.zeros((3,2))
    for i in range(3):
        current_coordinate=p[:,i]
        max_coordinate=np.max(current_coordinate)
        min_coordinate=np.min(current_coordinate)
        current_coordinate=(current_coordinate-min_coordinate)/(max_coordinate-min_coordinate)
        p[:,i]=current_coordinate
        coordinates_range[i,0]=max_coordinate
        coordinates_range[i,1]=min_coordinate
    return p,coordinates_range

"""
Berstein_function is used to compute the value of entry in Bernstein polynomial

Input: 
    b_order: A [i,j,k] Bernstein polynomial order in this entry for coordination (x,y,z).
    B_{N}^{i}(x)B_{N}^{j}(y)B_{N}^{k}(z) for this entry
    
    N: the maximal order of this Bernstein polynomial
    
    us: coordination (x,y,z) for this entry
    
OUTPUT:
    berstein_value: the corresponding value of this entry
    
"""  
def Berstein_function(b_order,N,us):
    berstein_value=1
    for i in range(3):
        berstein_value=berstein_value*comb(N,b_order[i])*((1-us[i])**(N-b_order[i]))*((us[i])**(b_order[i]))
        
    return berstein_value

"""
Input: 
    file_path1: where to get the output C_{expect} data, please note that considering the 
    size of each data set we do not return them for the following problems but load them again
    from the file_path1
    
    file_path2:where to load the 2021 PA 1-2 Student Data
    
    function_status: Two modes for datasets debug and unknown
    
    file_dictionary: lists of dataset to be processed
    
    b_order: The Bernstein polynomial order we want to fit in this function.
    The default value b_order=5.
    
OUTPUT:
    No direct returns,in SAVE_FILE_PATH Bernstein polynomial coefficient is saved in the 
    save_file_path by '_function_coeff.npy.npy' and the corresponding bounding range for each dataset is saved
    in '_coordinate_range.npy'. 
"""

def distortion_correction_Bernstein(file_path1,file_path2,function_status,dictionary,b_order=5):
################################ DATA LOADING PART ###############################
    calreading_name='-calreadings.txt'
    expect_name='_C_expect.npy'
    save_file_name='_function_coeff.npy'
    save_coordinate_range_name='_coordinate_range.npy'
    for file_name in dictionary:
        C_expect=np.load(file_path1+file_name+expect_name)        
        n_frame=C_expect.shape[0]
        C_expect=C_expect.transpose(0,2,1)
        C=[]
        for frame in range(n_frame):
            C.append(np.loadtxt(file_path2+function_status+'-'+file_name+calreading_name,skiprows=1,delimiter=',')[(16+(8+8+27)*frame):(43+(8+8+27)*frame),:])
        C=np.array(C)
        C_expect=C_expect.reshape(C_expect.shape[0]*C_expect.shape[1],-1)
        C=C.reshape(C.shape[0]*C.shape[1],-1)
#################### COMPUTATION OF BERNSTEIN MATRIX ##########################        
        C,coordinates_range=scale_to_box(C)
        Bernstein_matrix=np.zeros((C.shape[0],b_order+1,b_order+1,b_order+1))
        for s in range(C_expect.shape[0]):
            for i in range(b_order+1):
                for j in range (b_order+1):
                    for k in range(b_order+1):
                        Bernstein_matrix[s,i,j,k]=Berstein_function([i,j,k],b_order,C[s,:])
        
        Bernstein_matrix=Bernstein_matrix.reshape(C.shape[0],-1)  
        
######################### FIT THE BERNSTEIN POLYNOMIAL COEFFCIENTS########################        
        Cfit, residuals, _, _ = np.linalg.lstsq( Bernstein_matrix, C_expect , rcond=None)
######################### SAVING FILES  ##################################        
        save_path=file_path1+file_name+save_file_name
        np.save(save_path,Cfit)
        save_path2=file_path1+file_name+save_coordinate_range_name
        np.save(save_path2,coordinates_range)
        print('error', ((np.matmul(Bernstein_matrix,Cfit)-C_expect)**2).mean(axis=0))

#################FOR Q3~Q6 TO GET THE CORRECTED VALUE FOR G################
        
"""
bounding_box_with_known is used to normalized the g data between [0,1] with bounds from C

Input: 
    data: Nx3 matrix corresponding to (x,y,z) coordinates, which is normalized one by one.
    
    coordinate_range:The coordination range for (x,y,z) obtained from C. In this 3x2 matrix each
    row, the first entry is the maximal value and the second is the minimal value.
OUTPUT:
    data:The normalized coordination matrix
        
"""
def bounding_box_with_known(data,coordinate_range):
    for i in range(3):
        current_coordinate=data[:,i]
        max_coordinate=coordinate_range[i,0]
        min_coordinate=coordinate_range[i,1]

        current_coordinate=(current_coordinate-min_coordinate)/(max_coordinate-min_coordinate)
        data[:,i]=current_coordinate
        
    return data
"""
Input: 
    file_path1: where to get Bernstein polynomial data and bounding ranges, please note 
    that considering the size of each data set we do not return them for the following problems 
    but load them again from the file_path1
    
    file_path2:where to load the 2021 PA 1-2 Student Data
    
    function_status: Two modes for datasets debug and unknown
    
    file_dictionary: lists of dataset to be processed
    
    b_order: The Bernstein polynomial order in the used function.
    The default value b_order=5.
    
    input_file_name: corresponding distorted data's name such as for the empivot,fiducial and nav:
    input_file_name='-empivot.txt'; input_file_name='-em-fiducialss.txt' ; input_file_name='-EM-nav.txt'
    default value is input_file_name='-empivot.txt'
    
    G_save_name: corresponding save name for each output data, such as in PA2 for empivot,fiducial and nav:
    G_save_name='_G_empivot.npy'; G_save_name='_G_em_fiducials.npy'; G_save_name='_G_em_nav.npy'
    
OUTPUT:
    No direct returns,in SAVE_FILE_PATH corresponding corrected Gs are saved in end with G_save_name
    
"""        
def distortion_correction_output(file_path1,file_path2,function_status,dictionary,b_order=5, input_file_name='-empivot.txt',G_save_name='_G_empivot.npy'):
############## DATA LOADING ####################
       save_file_name='_function_coeff.npy'
       save_coordinate_range_name='_coordinate_range.npy'
       
       for file_name in dictionary:
           G=np.loadtxt(file_path2+function_status+'-'+file_name+input_file_name,skiprows=1,delimiter=',')
           function_coeff=np.load(file_path1+file_name+save_file_name)
           coordinate_range=np.load(file_path1+file_name+save_coordinate_range_name)
###################### BERNSTEIN POLYNOMIAL COMPUTATION#########################
           G=bounding_box_with_known(G,coordinate_range)
           Bernstein_matrix=np.zeros((G.shape[0],b_order+1,b_order+1,b_order+1))
           for s in range(G.shape[0]):
                for i in range(b_order+1):
                    for j in range (b_order+1):
                        for k in range(b_order+1):
                            Bernstein_matrix[s,i,j,k]=Berstein_function([i,j,k],b_order,G[s,:])
           Bernstein_matrix=Bernstein_matrix.reshape(G.shape[0],-1)
           G_calibration=np.matmul(Bernstein_matrix,function_coeff)
###################### SAVING FILES#########################
           
           np.save(file_path1+file_name+G_save_name,G_calibration)
           
           
"""
The following four functions are used to solved the practical problems from Q3~Q6. 
The input of these four functions are the same and the mainly difference is the inner
inputed variables based on the goal of the functions.
 
Input: 
    file_path1: The path to load and save the middle result of the DEMO and also all of the
    output files required in the program.
    
    file_path2:where to load data from the 2021 PA 1-2 Student Data
    
    function_status: Two modes for datasets debug and unknown
    
    file_dictionary: lists of dataset to be processed
        
OUTPUT:
    No direct returns, we save the corresponding variables required in the Questions or the variables
    need to be utilized for the following problems.
    
    Here is a list of the meaning of saved files from each of the following 4 functions:
    
    frame_pivot_calibration:
        1 save the result p_pivot and p_tip in NAME_p_pivot.npy, this is 2 3x1 vectors in a file
        2 save the corresponding g from the first frame to fix the coordinates relative to the probe
          in 'NAME_g_em.npy'
    
    fiducial_coordinate_em:
        save the obtained position of B_j in 'Name_B_em.npy'
    
    F_reg:
        save the computed frame between B and b in 'NAME_F_reg.npy'
        
    EM_CT_transformation:
        save the final output for the points in CT coordinates in 'NAME_G_ct_nav.npy'
"""
           
def frame_pivot_calibration(file_path1,file_path2,function_status,dictionary):
############## DATA LOADING ####################
    G_save_name='_G_empivot.npy'
    pivot_save_name='_p_pivot.npy'
    g_save_name='_g_em.npy'
    for file_name in dictionary:
           G_empivot=np.load(file_path1+file_name+G_save_name)
           
#############THE CALIBRATION IS BASED ON PA1 Q3 AND Q5#######################
           frames,g=PAf1.G_registeration(G_empivot)
           pfit=PAf1.pivot_calibration(frames)
           
################## SAVING FILES PARTS #########################
           np.save(file_path1+file_name+pivot_save_name,pfit)
           np.save(file_path1+file_name+g_save_name,g)
           
           
def fiducial_coordinate_em(file_path1,file_path2,function_status,dictionary):
############## DATA LOADING ####################
    G_save_name='_G_em_fiducials.npy'
    pivot_save_name='_p_pivot.npy'
    B_save_name='_B_em.npy'
    g_save_name='_g_em.npy'
    for file_name in dictionary:       
       G_fiducials=np.load(file_path1+file_name+G_save_name)
       g=np.load(file_path1+file_name+g_save_name)
       B=np.zeros((G_fiducials.shape[0]//6,3))
       p_tip=np.load(file_path1+file_name+pivot_save_name)[0:3].reshape(3,1)
       
#############THIS IS BASED ON PA1 FOR FRAME TRANSFORMATION#######################
       frames,g=PAf1.G_registeration(G_fiducials,g=g)
       for i in range (frames.shape[0]//3):
           B[i,:]=PAf1.frame_transformation(frames[3*i:3*(i+1),0:3],frames[3*i:3*(i+1),3],p_tip).reshape(3)

################## SAVING FILES PARTS #########################           
       np.save(file_path1+file_name+B_save_name,B)
       
       
def F_reg(file_path1,file_path2,function_status,dictionary):
############## DATA LOADING ####################
    B_save_name='_B_em.npy'
    b_input_name='-ct-fiducials.txt'
    F_reg_name='_F_reg.npy'
    for file_name in dictionary: 
        B=np.load(file_path1+file_name+B_save_name)
        b=np.loadtxt(file_path2+function_status+'-'+file_name+b_input_name,skiprows=1,delimiter=',')
#############THIS IS BASED ON PA1 FOR POINT REGISTERATION IN PA1 #######################
        F_reg=PAf1.points_registeration(B,b)
        
################## SAVING FILES PARTS ######################### 
        np.save(file_path1+file_name+F_reg_name,F_reg)
        
def EM_CT_transformation(file_path1,file_path2,function_status,dictionary):
############## DATA LOADING ####################     
    F_reg_name='_F_reg.npy'
    G_save_name='_G_em_nav.npy'
    G_output_name='_G_ct_nav.npy'
    g_save_name='_g_em.npy'
    pivot_save_name='_p_pivot.npy'    
    for file_name in dictionary: 
        F_reg=np.load(file_path1+file_name+F_reg_name)
        G_nav=np.load(file_path1+file_name+G_save_name)
        g=np.load(file_path1+file_name+g_save_name)
        p_tip=np.load(file_path1+file_name+pivot_save_name)[0:3].reshape(3,1)
#############THIS IS BASED ON PA1 #######################        
        frames,g=PAf1.G_registeration(G_nav,g=g)
        G_em_nav=np.zeros((G_nav.shape[0]//6,3))
        for i in range (frames.shape[0]//3):
           G_em_nav[i,:]=PAf1.frame_transformation(frames[3*i:3*(i+1),0:3],frames[3*i:3*(i+1),3],p_tip).reshape(3)
        G_CT=np.zeros(( G_em_nav.shape[0],3))
        for i in range( G_em_nav.shape[0]):
            G_CT[i,:]=PAf1.frame_transformation(F_reg[:,0:3],F_reg[:,3],G_em_nav[i,:].reshape(3,1)).reshape(3)
################## SAVING FILES PARTS ######################### 
        np.save(file_path1+file_name+G_output_name,G_CT)