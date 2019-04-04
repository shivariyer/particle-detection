from __future__ import division, absolute_import, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import cv2
import numpy as np
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt
from skimage import color
from monodepth_model import *
from monodepth_dataloader import *
from average_gradients import *
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import pearsonr

class Channel_value:
    val = -1.0
    intensity = -1.0

def find_intensity_of_atmospheric_light(img, gray):
    top_num = int(img.shape[0] * img.shape[1] * 0.001)
    toplist = [Channel_value()] * top_num
    dark_channel = find_dark_channel(img)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            val = img.item(y, x, dark_channel)
            intensity = gray.item(y, x)
            for t in toplist:
                if t.val < val or (t.val == val and t.intensity < intensity):
                    t.val = val
                    t.intensity = intensity
                    break

    max_channel = Channel_value()
    for t in toplist:
        if t.intensity > max_channel.intensity:
            max_channel = t

    return max_channel.intensity

def find_dark_channel(img):
    return np.unravel_index(np.argmin(img), img.shape)[2]

def haze_1d(img, light_intensity, windowSize, t0, w):
    size = (img.shape[0], img.shape[1])
    outimg = np.zeros(size, img.dtype)

    for y in range(size[0]):
        for x in range(size[1]):
            x_low = max(x-(windowSize//2), 0)
            y_low = max(y-(windowSize//2), 0)
            x_high = min(x+(windowSize//2), size[1])
            y_high = min(y+(windowSize//2), size[0])
            sliceimg = img[y_low:y_high, x_low:x_high]
            dark_channel = find_dark_channel(sliceimg)
            t = 1.0 - (w * img.item(y, x, dark_channel) / light_intensity)
            outimg.itemset((y,x), t)
            #outimg.itemset((y,x), max(t,t0)*255)
    
    img_arr = np.ravel(outimg)

    return img_arr

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def depth_1d(params,image_path,checkpoint_path,city):

    left  = tf.placeholder(tf.float32, [2, 256, 256, 3])
    model = MonodepthModel(params, "test", left, None)

    input_image = scipy.misc.imread(image_path, mode="RGB")
    
    if city == "B":
        input_image = input_image[:650, :]
    else:
        input_image = input_image[50:350, :]
        
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [256, 256], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = checkpoint_path #.split(".")[0]
    train_saver.restore(sess, restore_path)

    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

    #output_directory = os.path.dirname(image_path)
    #output_name = os.path.splitext(os.path.basename(image_path))[0]

    #np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    disp_to_img = color.rgb2gray(disp_to_img)
    #plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')
    img_arr = np.ravel(disp_to_img)

    return img_arr

def main():
	data = pd.read_csv('../single_data.csv')
	params = monodepth_parameters(encoder='vgg',height=256,width=256,batch_size=2,num_threads=1,num_epochs=1,do_stereo=False,wrap_mode="border",use_deconv=False,alpha_image_loss=0,disp_gradient_loss_weight=0,lr_loss_weight=0,full_summary=False)

	pm = []
	per50 = []
	per75 = []
	per90 = []
	permean = []
	permax = []

	for index, row in data.iterrows():
	    city = row['filename'][0]
	    filename = '../'+row['filename']+'.jpg'
	    if city == "B":
	        img = cv2.imread(filename)[:650, :]
	        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	        light_intensity = find_intensity_of_atmospheric_light(img, gray)
	        w = 0.9
	        t0 = 0.01
	        haze_arr = haze_1d(img, light_intensity, 20, t0, w)
	        depth_arr = depth_1d(params,filename,'../city2kitti/model_city2kitti',city)
	        mult_arr = np.multiply(depth_arr, haze_arr)

	        per50.append(np.percentile(mult_arr,50))
	        per75.append(np.percentile(mult_arr,75))
	        per90.append(np.percentile(mult_arr,90))
	        permean.append(np.average(mult_arr))
	        permax.append(np.max(mult_arr))
	        pm.append(float(row['ppm']))
	        print(index)
	    # else:
	    #     img = cv2.imread(filename)[50:350, :]

if __name__ == '__main__':
	tf.app.run()
