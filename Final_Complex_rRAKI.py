import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import scipy.io as sio
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

matfn = '/content/drive/My Drive/ICASSP_rRAKI/data/multicoil_train/multicoil_train/file1000142.mat'
name_weight = '/content/drive/My Drive/ICASSP_rRAKI/Complex_rRAKI_Results/Knee/file1000142weight.mat'
recon_name = '/content/drive/My Drive/ICASSP_rRAKI/Complex_rRAKI_Results/Knee/file1000142recon.mat'

# matfn = '/content/drive/My Drive/ICASSP_rRAKI/data/brain_multicoil_train/brain_multicoil_train/file_brain_AXFLAIR_200_6002466.mat'
# name_weight = '/content/drive/My Drive/ICASSP_rRAKI/Complex_rRAKI_Results/Brain/file_brain_AXFLAIR_200_6002466weight.mat'
# recon_name = '/content/drive/My Drive/ICASSP_rRAKI/Complex_rRAKI_Results/Brain/file_brain_AXFLAIR_200_6002466recon.mat'

def complex_to_channels(image):            
    """Convert data from complex to channels."""
    image_out = tf.stack([tf.math.real(image), tf.math.imag(image)], axis=-1)
    shape_out = tf.concat(
        [tf.shape(image)[:-1], [image.shape[-1] * 2]], axis=0)
    image_out = tf.reshape(image_out, shape_out)
    return image_out

def weight_variable(shape,vari_name):                   
    initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
    return tf.Variable(initial,name = vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)
    
# def zrelu(x):   #fire only in first quadrant
#     shape_x = x.shape  #(1, 640, 25, 32)
#     s = int(0 if shape_x[-1] is None else shape_x[-1])
    
#     def _compare_(phase,x):
#         ind1 = tf.where(phase>=0)
#         ind =  tf.where(phase[ind1]<=((math.pi)/2))
#         x_z = tf.zeros(shape_x)
#         x_z[ind] = x[ind]
#         return x_z
        
#     real_x = x[:,:,:,:s//2]
#     imag_x = x[:,:,:,s//2:]
#     cplx = tf.complex(real_x,imag_x)
#     mag = tf.math.abs(cplx)
#     phase = tf.math.angle(cplx)
#     return tf.py_function(
#         _compare_,
#         [phase,x],
#         tf.float32,
#     )
    
        
# bmr = tf.Variable(tf.constant(1.0, dtype=tf.float32))
        
# def modrelu(x):   #fire outside a circle about the origin
#     shape_x = x.shape  #(1, 640, 25, 32)
#     real_x = x[:,:,:,:shape_x[-1]//2]
#     imag_x = x[:,:,:,shape_x[-1]//2:]
#     cplx = tf.complex(real_x,imag_x)
#     mag = tf.math.abs(cplx)
#     phase = tf.math.angle(cplx)
#     if(mag+bmr>=0):
#         return (tf.multiply((mag+bmr),x))/mag
#     else:
#         return tf.zeros(shape_x)
        
def init_Newrelu_coefficients():
    coeff_a = tf.Variable(tf.truncated_normal(([1]), stddev=0.1,dtype=tf.float32))
    coeff_b = tf.Variable(tf.truncated_normal(([1]), stddev=0.1,dtype=tf.float32))
    coeff_c = tf.Variable(tf.truncated_normal(([1]), stddev=0.1,dtype=tf.float32))
    # print("INIT act coefficients ", sess.run(coeff_a),sess.run(coeff_b),sess.run(coeff_c))
    return coeff_a,coeff_b,coeff_c
        
def Newrelu(x,coeff_a,coeff_b,coeff_c):
    shape_x = x.shape  
    s = int(0 if shape_x[-1] is None else shape_x[-1])
    real_x = x[:,:,:,:s//2]
    imag_x = x[:,:,:,s//2:]
    c_x = tf.complex(real_x, imag_x)
    dummy = tf.zeros(real_x.shape,dtype=tf.complex64)
    result = tf.where((coeff_a*tf.math.real(c_x)+coeff_b*tf.math.imag(c_x)+coeff_c)>0,c_x,dummy)
    return complex_to_channels(result)
    

def cconv2d_dilate(x, real_W, imag_W, dilate_rate):
    shape_x = x.shape  #(1, 640, 25, 32)
    s = int(0 if shape_x[-1] is None else shape_x[-1])
    real_x = x[:,:,:,:s//2]
    imag_x = x[:,:,:,s//2:]

    real_real = tf.nn.convolution(real_x, real_W, padding='VALID',dilation_rate = [1,dilate_rate])
    real_imag = tf.nn.convolution(real_x, imag_W, padding='VALID',dilation_rate = [1,dilate_rate])
    imag_real = tf.nn.convolution(imag_x, real_W, padding='VALID',dilation_rate = [1,dilate_rate])
    imag_imag = tf.nn.convolution(imag_x, imag_W, padding='VALID',dilation_rate = [1,dilate_rate])   

    out_real = real_real-imag_imag
    out_imag = imag_real+real_imag

    complex_output = tf.complex(out_real, out_imag)

    channels_output = complex_to_channels(complex_output)

    return channels_output
    

#### LEARNING FUNCTION ####
def learning(accrate_input,sess):
    # define placeholder for inputs to network
    
    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])                                   
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_Z])           # target size
    
    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])         

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)

    ker_conv_r = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z//2, target_dim3//2],'G1')
    ker_conv_i = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z//2, target_dim3//2],'G2')
    grp_conv = cconv2d_dilate(Input, ker_conv_r, ker_conv_i ,accrate_input)

    x_shift = np.int32(np.floor(kernel_last_x/2))
    
    [aa,bb,dim_yy,cc]=np.shape(grp_conv);

    grap_y_start = np.int32(  (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * accrate_input
    grap_y_end = np.int32(dim_yy) - np.int32(( (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * accrate_input - 1


    grap_y_start = np.int32(grap_y_start);
    grap_y_end = np.int32(grap_y_end+1);
    

    grapRes =  grp_conv[:,x_shift:x_shift+target_dim_X,grap_y_start:grap_y_end,:];
    W_conv1_r = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z//2, layer1_channels//2],'W11')
    W_conv1_i = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z//2, layer1_channels//2],'W12') 
    # h_conv1 = tf.nn.relu(cconv2d_dilate(Input, W_conv1_r, W_conv1_i, accrate_input)) 
    
    coeff_a1,coeff_b1,coeff_c1 = init_Newrelu_coefficients()
    h_conv1 = Newrelu(cconv2d_dilate(Input, W_conv1_r, W_conv1_i, accrate_input), coeff_a1,coeff_b1,coeff_c1)


    ## conv2 layer
    W_conv2_r = weight_variable([kernel_x_2, kernel_y_2, layer1_channels//2, layer2_channels//2],'W21')
    W_conv2_i = weight_variable([kernel_x_2, kernel_y_2, layer1_channels//2, layer2_channels//2],'W22')
    # h_conv2 = tf.nn.relu(cconv2d_dilate(h_conv1, W_conv2_r, W_conv2_i, accrate_input))
    
    coeff_a2,coeff_b2,coeff_c2 = init_Newrelu_coefficients()
    h_conv2 = Newrelu(cconv2d_dilate(h_conv1, W_conv2_r, W_conv2_i, accrate_input), coeff_a2,coeff_b2,coeff_c2)

    ## conv3 layer
    W_conv3_r = weight_variable([kernel_last_x, kernel_last_y, layer2_channels//2, target_dim3//2],'W31')
    W_conv3_i = weight_variable([kernel_last_x, kernel_last_y, layer2_channels//2, target_dim3//2],'W32')
    h_conv3 = cconv2d_dilate(h_conv2, W_conv3_r, W_conv3_i, accrate_input)

    error_norm = 1*tf.norm(input_Target - grapRes - h_conv3) + 1*tf.norm(input_Target - grapRes) 
    print("MOMENTUM LEARNING")
    train_step = tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.9,use_nesterov=False).minimize(error_norm)
    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1  # initial error, we set 1 and it began to decrease.
    for i in range(2000):
        
        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target})
        if i % 100 == 0:                                                                         
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})     
            print('The',i,'th iteration gives an error',error_now)                              
            
    error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    print("LEARNT act coefficients ", sess.run(coeff_a1),sess.run(coeff_b1),sess.run(coeff_c1), sess.run(coeff_a2),sess.run(coeff_b2),sess.run(coeff_c2))
    return [sess.run(ker_conv_r),sess.run(ker_conv_i),sess.run(W_conv1_r),sess.run(W_conv1_i),sess.run(W_conv2_r),sess.run(W_conv2_i),sess.run(W_conv3_r),sess.run(W_conv3_i),sess.run(coeff_a1),sess.run(coeff_b1),sess.run(coeff_c1),sess.run(coeff_a2),sess.run(coeff_b2),sess.run(coeff_c2),error]   
    
                                                            #### RECON CONVOLUTION FUNCTION ####
def cnn_3layer(input_kspace,gkerr,gkeri,w1r,w1i,b1,w2r,w2i,b2,w3r,w3i,b3,acc_rate,coeff_a1,coeff_b1,coeff_c1,coeff_a2,coeff_b2,coeff_c2,sess):                 
    grap = cconv2d_dilate(input_kspace, gkerr, gkeri ,acc_rate)
    x_shift = np.int32(np.floor(kernel_last_x/2))
    
    [aa,dim_x,dim_yy,dd] = np.shape(grap);

    grap_y_start = np.int32((np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate
    grap_y_end = np.int32(dim_yy) - np.int32(( (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate - 1

    grap_y_start = np.int32(grap_y_start);
    grap_y_end = np.int32(grap_y_end+1);

    effectiveGrappa =  grap[:, x_shift:dim_x-x_shift, grap_y_start:grap_y_end, :];

    h_conv1 = Newrelu(cconv2d_dilate(input_kspace, w1r, w1i, acc_rate), coeff_a1,coeff_b1,coeff_c1) 
    h_conv2 = Newrelu(cconv2d_dilate(h_conv1, w2r, w2i, acc_rate), coeff_a2,coeff_b2,coeff_c2)
    h_conv3 = cconv2d_dilate(h_conv2, w3r, w3i, acc_rate) 
    
    print('grap res shape = ',np.shape(grap))
    print('effective grap shape = ',np.shape(effectiveGrappa))
    print('h_conv shape = ',np.shape(h_conv3))
    return sess.run(effectiveGrappa+h_conv3), sess.run(effectiveGrappa),sess.run(h_conv3), sess.run(grap) 


                                            ######### ----- INITIALIZE ----- ###########


no_ACS_flag = 0;

rate = 5
ACSrange = 24
phaseshiftflag = 0;
kspace = sio.loadmat(matfn)
kspace = kspace['kspace'] # get kspace
mask = np.zeros_like(kspace)
[row,col,coil] = mask.shape
mask[:,::rate,:] = 1;
midpoi = col//2;
mask[:,midpoi-ACSrange//2+1:midpoi+ACSrange//2,:]=1
kspace = kspace*mask
normalize = 1/np.max(abs(kspace[:]))
kspace = np.multiply(kspace,normalize)   

[m1,n1,no_ch] = np.shape(kspace)# for no_inds = 1 here
no_inds = 1

kspace_all = kspace;
kx = np.transpose(np.int32([(range(1,m1+1))]))                          
ky = np.int32([(range(1,n1+1))])

if phaseshiftflag ==1:
    phase_shifts = np.dot(np.exp(-1j * 2 * 3.1415926535 / m1 * (m1/2-1) * kx ),np.exp(-1j * 2 * 3.14159265358979 / n1 * (n1/2-1) * ky ))
    for channel in range(0,no_ch):
        kspace_all[:,:,channel] = np.multiply(kspace_all[:,:,channel],phase_shifts)


kspace = np.copy(kspace_all)
mask = np.squeeze(np.sum(np.sum(np.abs(kspace),axis=0),axis=1))>0
picks = np.where(mask == 1);                                  
kspace = kspace[:,np.int32(picks[0][0]):n1+1,:]
kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]  # this part erase the all zero columns before the 1st sampled column

kspace_NEVER_TOUCH = np.copy(kspace_all)

mask = np.squeeze(np.sum(np.sum(np.abs(kspace),axis=0),axis=1))>0;  
picks = np.where(mask == 1);                                  
d_picks = np.diff(picks,1)  # this part finds the ACS region. if no diff==1, means no continuous sample lines, then no_ACS_flag==1
indic = np.where(d_picks == 1);

mask_x = np.squeeze(np.sum(np.sum(np.abs(kspace),axis=2),axis=1))>0;
picks_x = np.where(mask_x == 1);
x_start = picks_x[0][0]
x_end = picks_x[0][-1]

if np.size(indic)==0:    # if there is no no continuous sample lines, it means no ACS in the input
    no_ACS_flag=1;       # set flag
    print('No ACS signal in input data, using individual ACS file.')
    matfn = 'data_diffusion/ACS_29.mat'    # read outside ACS in 
    matfn = 'RO6/ACS6.mat'
    ACS = sio.loadmat(matfn)
    ACS = ACS['ACS']     
    normalize = 0.015/np.max(abs(ACS[:])) # Has to be the same scaling or it won't work
    ACS = np.multiply(ACS,normalize*scaling)

    [m2,n2,no_ch2] = np.shape(ACS)# for no_inds = 1 here
    no_inds = 1

    kx2 = np.transpose(np.int32([(range(1,m2+1))]))                          # notice here 1:m1 = 1:m1+1 in python
    ky2 = np.int32([(range(1,n2+1))])
    
    if phaseshiftflag ==1:
        phase_shifts = np.dot(np.exp(-1j * 2 * 3.1415926535 / m2 * (m2/2-1) * kx2 ),np.exp(-1j * 2 * 3.14159265358979 / n2 * (n2/2-1) * ky2 ))
        for channel in range(0,no_ch2):
            ACS[:,:,channel] = np.multiply(ACS[:,:,channel],phase_shifts)

    kspace = np.multiply(kspace,scaling)
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    print(ACS_dim_X)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)
else:
    no_ACS_flag=0;
    print('ACS signal found in the input data')
    indic = indic[1][:]
    center_start = picks[0][indic[0]];
    center_end = picks[0][indic[-1]+1];

    print('START AND END OF ACS ',center_start,center_end)

    ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)


acc_rate = d_picks[0][0]
no_channels = ACS_dim_Z*2


kernel_x_1 = 5
kernel_y_1 = 2

kernel_x_2 = 1
kernel_y_2 = 1

kernel_last_x = 3
kernel_last_y = 2

layer1_channels = 32
layer2_channels = 8


gker_all_r = np.zeros([kernel_x_1, kernel_y_1, no_channels//2, acc_rate - 1, no_channels//2],dtype=np.float32)
w1_all_r = np.zeros([kernel_x_1, kernel_y_1, no_channels//2, layer1_channels//2, no_channels//2],dtype=np.float32)
w2_all_r = np.zeros([kernel_x_2, kernel_y_2, layer1_channels//2,layer2_channels//2,no_channels//2],dtype=np.float32)
w3_all_r = np.zeros([kernel_last_x, kernel_last_y, layer2_channels//2,acc_rate - 1, no_channels//2],dtype=np.float32)    

gker_all_i = np.zeros([kernel_x_1, kernel_y_1, no_channels//2, acc_rate - 1, no_channels//2],dtype=np.float32)
w1_all_i = np.zeros([kernel_x_1, kernel_y_1, no_channels//2, layer1_channels//2, no_channels//2],dtype=np.float32)
w2_all_i = np.zeros([kernel_x_2, kernel_y_2, layer1_channels//2,layer2_channels//2,no_channels//2],dtype=np.float32)
w3_all_i = np.zeros([kernel_last_x, kernel_last_y, layer2_channels//2, acc_rate - 1, no_channels//2],dtype=np.float32)

b1_flag = 0;
b2_flag = 0;                         
b3_flag = 0;

if (b1_flag == 1):
    b1_all = np.zeros([1,1, layer1_channels,no_channels]);
else:
    b1 = []

if (b2_flag == 1):
    b2_all = np.zeros([1,1, layer2_channels,no_channels])
else:
    b2 = []

if (b3_flag == 1):
    b3_all = np.zeros([1,1, layer3_channels, no_channels])
else:
    b3 = []


target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1);  # remember every indicies need to -1 in python
target_x_end = np.int32(ACS_dim_X - target_x_start -1)

################################ ----- initialize done, lets RAKI! ----- #######################################

time_ALL_start = time.time()

################################ ----- LEARNING PART! ----- #######################################

[ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z]) # Batch, X, Y, Z
ACS = np.float32(ACS)  # here we use ACS instead of ACS_re for convinience


target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate;     #-------------------------------------
target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1;

target_dim_X = target_x_end - target_x_start + 1
target_dim_Y = target_y_end - target_y_start + 1
target_dim_Z = (acc_rate - 1)*2 # *2 for complex valued 

print('go!')
time_Learn_start = time.time() # set timer

errorSum = 0;
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1/3; # avoid fully allocating.


for ind_c in range(ACS_dim_Z//2):
    sess = tf.Session(config=config)  
    target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])
    print('learning channel #',ind_c+1)
    time_channel_start = time.time()
    lim = acc_rate-1 
    for ind_acc in range(lim):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1
        target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
        target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];
        target[0,:,:,ind_acc+lim] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c+no_channels//2];

    [gkerr,gkeri,w1r,w1i,w2r,w2i,w3r,w3i,error,coeff_a1,coeff_b1,coeff_c1,coeff_a2,coeff_b2,coeff_c2]=learning(acc_rate,sess) 
    gker_all_r[:,:,:,:,ind_c]=gkerr
    w1_all_r[:,:,:,:,ind_c] = w1r
    w2_all_r[:,:,:,:,ind_c] = w2r
    w3_all_r[:,:,:,:,ind_c] = w3r   

    gker_all_i[:,:,:,:,ind_c]=gkeri
    w1_all_i[:,:,:,:,ind_c] = w1i
    w2_all_i[:,:,:,:,ind_c] = w2i
    w3_all_i[:,:,:,:,ind_c] = w3i 
    
    time_channel_end = time.time()
    print('Time Cost:',time_channel_end-time_channel_start,'s')
    print('Norm of Error = ',error)
    errorSum = errorSum + error

    sess.close()
    tf.reset_default_graph()

time_Learn_end = time.time();
print('lerning step costs:',(time_Learn_end - time_Learn_start)/60,'min') # get time

sio.savemat(name_weight, {'gkerr':gker_all_r,'w1r': w1_all_r,'w2r': w2_all_r,'w3r': w3_all_r, 'gkeri':gker_all_i,'w1i': w1_all_i,'w2i': w2_all_i,'w3i': w3_all_i})  


weightfile = sio.loadmat(name_weight)
gker_allr = weightfile['gkerr'] # get kspace
gker_alli = weightfile['gkeri'] # get kspace
w1_allr = weightfile['w1r'] # get kspace
w1_alli = weightfile['w1i'] # get kspace
w2_allr = weightfile['w2r'] # get kspace
w2_alli = weightfile['w2i'] # get kspace
w3_allr = weightfile['w3r'] # get kspace
w3_alli = weightfile['w3i'] # get kspace


                                                        ################ ----- RECON PART! ----- ################


kspace_recon_all = np.copy(kspace_all)
kspace_recon_all_nocenter = np.copy(kspace_all)

kspace = np.copy(kspace_all)

over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))
kspace_und = kspace
kspace_und[:,over_samp,:] = 0;
[dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
kspace_und_re = np.float32(kspace_und_re)
kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
kspace_recon = kspace_und_re

raki_recon = np.zeros_like(kspace_recon)
grap_recon = np.copy(kspace_recon)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1/3 ; # avoid fully allocating.


for ind_c in range(0,no_channels//2):
    print('Reconstruting Channel #',ind_c+1)


    sess = tf.Session(config=config)  # tensorflow initialize
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)


    # grab w and b
    gkerr = np.float32(gker_allr[:,:,:,:,ind_c])
    gkeri = np.float32(gker_alli[:,:,:,:,ind_c])
    w1r = np.float32(w1_allr[:,:,:,:,ind_c])
    w1i = np.float32(w1_alli[:,:,:,:,ind_c])
    w2r = np.float32(w2_allr[:,:,:,:,ind_c])
    w2i = np.float32(w2_alli[:,:,:,:,ind_c])
    w3r = np.float32(w3_allr[:,:,:,:,ind_c])
    w3i = np.float32(w3_alli[:,:,:,:,ind_c])

    if (b1_flag == 1):
        b1 = b1_all[:,:,:,ind_c];
    if (b2_flag == 1):
        b2 = b2_all[:,:,:,ind_c];
    if (b3_flag == 1):
        b3 = b3_all[:,:,:,ind_c];                
        
    [res,grap,raki,rawgrap] = cnn_3layer(kspace_und_re,gkerr,gkeri,w1r,w1i,b1,w2r,w2i,b2,w3r,w3i,b3,acc_rate,coeff_a1,coeff_b1,coeff_c1,coeff_a2,coeff_b2,coeff_c2,sess)  
    target_x_end_kspace = dim_kspaceUnd_X - target_x_start;
    
    for ind_acc in range(0,acc_rate-1):

        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1
        target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc;
        kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = res[0,:,::acc_rate,ind_acc]

        grap_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = grap[0,:,::acc_rate,ind_acc]

        raki_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = raki[0,:,::acc_rate,ind_acc]


    print('total parameters ', np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    sess.close()
    tf.reset_default_graph()

kspace_recon = np.squeeze(kspace_recon)

kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex); # im_ind = 1, skip one dim


grap_recon = np.squeeze(grap_recon)

grap_recon_complex = (grap_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(grap_recon[:,:,np.int32(no_channels/2):no_channels],1j))

raki_recon = np.squeeze(raki_recon)

raki_recon_complex = (raki_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(raki_recon[:,:,np.int32(no_channels/2):no_channels],1j)) 


if no_ACS_flag == 0:  # if we have ACS region in kspace, put them back
    kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
    print('ACS signal has been put back')
else:
    print('No ACS signal is put into k-space')



kspace_recon_all[:,:,:] = kspace_recon_complex; # im_ind = 1, skip one dim

for sli in range(0,no_ch):
    kspace_recon_all[:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,sli])

rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
sio.savemat(recon_name,{'kspace_all':kspace_recon_complex,'kspace_all_noACS':kspace_recon_all_nocenter,'grap_all':grap_recon_complex,'raki_all':raki_recon_complex,'rawgrap':rawgrap,'effectiveGrap':grap})  # save the results

time_ALL_end = time.time()
print('All process costs ',(time_ALL_end-time_ALL_start)/60,'mins')
print('Error Average in Training is ',errorSum/no_channels)
# np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])