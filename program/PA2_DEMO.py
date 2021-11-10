# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 12:53:04 2021

@author: Ding
"""
import PA2_functions as PAf2
"""
General input of this demo to run the codes for PA2:
    
    file_dictionary: the list of dataset to processed
    example: file_dictionary=['g','h','i','j'], corresponding to the unknown data set part.
    
    folder_path: the original path of the 2021 PA 1-2 Student Data, please note that this path should
                end up with 'pa2-'.
    example:folder_path='.\\2021_pa_1-2_student_data\\2021 PA 1-2 Student Data\\pa2-'
    
    save_file_path: path to save the output,which is also used to load the mid result such as C_expect
    example:save_file_path='CIS\\PA2\\result\\'
    
    function_status: two modes debug and unknown for loading two kinds of dataset
    example:function_status='debug' function_status='unknown'
"""

##########################GENERAL INPUT PART################################
file_dictionary=['a','b','c','d','e','f']
folder_path='..\\PA2\\2021 PA 1-2 Student Data\\pa2-'
save_file_path='..\\PA2\\OUTPUT\\'
function_status='debug'

#################################QUESTION 1 PART###############################
"""
Input: 
    folder_path:where to load the 2021 PA 1-2 Student Data
    
    save_file_path: where to save the output C_{expect} data, please note that considering the 
    size of each data set we do not return them for the following problems but load them again
    from the SAVE_FILE_PATH
    
    function_status: Two modes for datasets debug and unknown
    
    file_dictionary: lists of dataset to be processed
    
OUTPUT:
    No direct returns, C_expect is saved in the save_file_path by '_C_expect.npy'
"""
PAf2.C_expect_output( folder_path,save_file_path,function_status,file_dictionary)

#####################QUESTION 2 PART FITTING DISTORTION CORRECTION FUNCTION#######################
"""
Input: 
    save_file_path: where to get the output C_{expect} data, please note that considering the 
    size of each data set we do not return them for the following problems but load them again
    from the SAVE_FILE_PATH
    
    folder_path:where to load the 2021 PA 1-2 Student Data
    
    function_status: Two modes for datasets debug and unknown
    
    file_dictionary: lists of dataset to be processed
    
    b_order: The Bernstein polynomial order we want to fit in this function.
    The default value b_order=5.
    
OUTPUT:
    No direct returns,in SAVE_FILE_PATH Bernstein polynomial coefficient is saved in the 
    save_file_path by '_function_coeff.npy.npy' and the corresponding bounding range for each dataset is saved
    in '_coordinate_range.npy'. 
"""
PAf2.distortion_correction_Bernstein(save_file_path,folder_path,function_status,file_dictionary,b_order=5)

########################QUESTION 3~6: THE OUTPUT FOR EACH G's##########################
"""
Input: 
    save_file_path: where to get Bernstein polynomial data and bounding ranges, please note 
    that considering the size of each data set we do not return them for the following problems 
    but load them again from the SAVE_FILE_PATH
    
    folder_path:where to load the 2021 PA 1-2 Student Data
    
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
###########EM PIVOT CORRECTION###############
PAf2.distortion_correction_output(save_file_path,folder_path,function_status,file_dictionary,b_order=5)
###########EM FIDUCIAL CORRECTION###############
PAf2.distortion_correction_output(save_file_path,folder_path,function_status,file_dictionary,b_order=5,input_file_name='-em-fiducialss.txt',G_save_name='_G_em_fiducials.npy')
###########EM TEST(NAV) CORRECTION###############
PAf2.distortion_correction_output(save_file_path,folder_path,function_status,file_dictionary,b_order=5,input_file_name='-EM-nav.txt',G_save_name='_G_em_nav.npy')

"""
The following four functions are used to solved the practical problems from Q3~Q6. 
The input of these four functions are the same and the mainly difference is the inner
inputed variables based on the goal of the functions.
 
Input: 
    save_file_path: The path to load and save the middle result of the DEMO and also all of the
    output files required in the program.
    
    folder_path:where to load data from the 2021 PA 1-2 Student Data
    
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

########## PIVOT CALIBRATION PART FOR Q3 #######################
PAf2.frame_pivot_calibration(save_file_path,folder_path,function_status,file_dictionary)

########## COMPUTE THE COORDINATES OF B(JUST Q4) #######################
PAf2.fiducial_coordinate_em(save_file_path,folder_path,function_status,file_dictionary)

########## COMPUTE THE F_REG,Q5  #######################
PAf2.F_reg(save_file_path,folder_path,function_status,file_dictionary)

########## PREDICT THE OUTPUT VALUE IN CT COORDINATES Q6 #######################
PAf2.EM_CT_transformation(save_file_path,folder_path,function_status,file_dictionary)

