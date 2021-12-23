import numpy as np
import cv2
import random
from scipy import ndimage
import math
from shape_adjustment import *

class transforms:
    
    def __init__(self):
        pass

    #Downscales the image by a user defined factor
    def downscale(self, input_img, coordinates, bg_img, scale=2):   
        if input_img.shape[0]%scale!=0:
            pad_bottom = input_img.shape[0]%scale
        else:
            pad_bottom = 0
        if input_img.shape[1]%scale!=0:
            pad_right = input_img.shape[1]%scale
        else:
            pad_right = 0
            
        rb = np.pad(input_img[:,:,0],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        gb = np.pad(input_img[:,:,1],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        bb = np.pad(input_img[:,:,2],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        input_img = np.dstack(tup=(rb, gb, bb)) 

        new_coordinates = [[i[0]/scale, i[1]/scale] for i in coordinates]
        aug_img =  input_img[::scale,::scale]

        return (aug_img, new_coordinates)
    
    #Upscales the image by a user defined factor
    def upscale(self, input_img, coordinates, bg_img, scale=2):    
        if input_img.shape[0]%scale!=0:
            pad_bottom = input_img.shape[0]%scale
        else:
            pad_bottom = 0
        if input_img.shape[1]%scale!=0:
            pad_right = input_img.shape[1]%scale
        else:
            pad_right = 0
            
        rb = np.pad(input_img[:,:,0],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        gb = np.pad(input_img[:,:,1],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        bb = np.pad(input_img[:,:,2],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        input_img = np.dstack(tup=(rb, gb, bb)) 

        new_coordinates = [[i[0]*scale, i[1]*scale] for i in coordinates]
        aug_img =  cv2.pyrUp(input_img, scale)
        
        return (aug_img, new_coordinates)

    # Rotates coordinates
    def rotation(self, input_img, coordinates, rotlimitangle):        
        temp_coordinates = np.asarray(coordinates)
        temp_coordinates = np.asarray([temp_coordinates[:,1],temp_coordinates[:,0]])

        alpha = np.random.randint(0,rotlimitangle+1)
        theta = alpha % 90

        cos_alpha = math.cos(math.radians(alpha))
        sin_alpha = math.sin(math.radians(alpha))
        cos_theta = math.cos(math.radians(theta))
        sin_theta = math.sin(math.radians(theta))

        bias = np.zeros(temp_coordinates.shape)
        bias[0] = input_img.shape[0]
        bias[1] = input_img.shape[1]

        if alpha>=0 and alpha<90:   
            R_b = np.array([[0, sin_theta],[0, 0]])   
        elif alpha>=90 and alpha<180:
            R_b = np.array([[sin_theta, cos_theta],[0, sin_theta]]) 
        elif alpha>=180 and alpha<270:
            R_b = np.array([[cos_theta, 0],[sin_theta, cos_theta]])
        elif alpha>=270 and alpha<360:    
            R_b = np.array([[0, 0],[cos_theta, 0]])

        R = np.array([[cos_alpha, -sin_alpha],[sin_alpha, cos_alpha]])

        new_matrix = np.dot(R, temp_coordinates) + np.dot(R_b, bias)
        rotated_coordinates = new_matrix[::-1].T.tolist()
        
        aug_img = ndimage.rotate(input_img, alpha, reshape=True)

        return (aug_img, rotated_coordinates)
    
    # Flips the image about horizontal axis
    def fliphorizontal(self, input_img, coordinates):        
        aug_img = input_img[::-1,::] 
        aug_coordinates = [[i[0], input_img.shape[0] - i[1]] for i in coordinates]

        return (aug_img, aug_coordinates)
    
    # Flips the image about vertical axis
    def flipvertical(self, input_img, coordinates):       
        aug_img = input_img[::,::-1]
        aug_coordinates = [[input_img.shape[1] - i[0], i[1]] for i in coordinates]

        return (aug_img, aug_coordinates)

    # Blur
    def blur(self,input_img, blurchoice):        
        blurtype = ['Averaging', 'Gaussian Blurring', 'Median Blurring', 'Bilateral Filtering']
        if(blurchoice!=0 and blurchoice!=1 and blurchoice!=2 and blurchoice!=3):
            ch = np.random.randint(0,4)
            blurchoice = ch
        if  blurchoice == 0:
            aug_img = cv2.blur(input_img, (2,2)) 
        elif  blurchoice == 1:
            aug_img = cv2.GaussianBlur(input_img,(5,5),0)
        elif  blurchoice == 2:
            aug_img = cv2.medianBlur(input_img,5)
        elif  blurchoice == 3:
            aug_img = cv2.bilateralFilter(input_img,9,75,75)
            
        return aug_img
    
    # Noise
    def noise(self, input_img, noise_choice, p_gauss =[0, 0.1], p_sp =[0.5, 0.004], p_speckle =[0.1]):              
        noise_type = ["gauss", "salt&pepper", "poisson", "speckle"]        
        if(noise_choice!=0 and noise_choice!=1 and noise_choice!=2 and noise_choice!=3):
            ch = np.random.randint(0,4)
            noise_choice = ch
            
        if noise_choice == 0:
            mean,variance = p_gauss[0], p_gauss[1]
            sigma = variance**0.5
            gauss = np.random.normal(mean,sigma,input_img.shape)
            aug_img = input_img + gauss
            aug_img = np.clip(aug_img, 0, 255)
        
        elif noise_choice == 1:
            s_vs_p, amount = p_sp[0], p_sp[1] 
            # Salt mode
            num_salt = np.ceil(amount * input_img.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in input_img.shape]
            input_img[tuple(coords)] = 255
            # Pepper mode
            num_pepper = np.ceil(amount* input_img.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in input_img.shape]
            input_img[tuple(coords)] = 0
            aug_img = input_img
        
        elif noise_choice == 2:
            vals = len(np.unique(input_img))
            vals = 2 ** np.ceil(np.log2(vals))
            aug_img = np.random.poisson(input_img * vals) / float(vals)
        
        elif noise_choice == 3:
            gauss = np.random.randn(input_img.shape[0], input_img.shape[1], input_img.shape[2])      
            aug_img = input_img + p_speckle[0]*(input_img * gauss)
            aug_img = np.clip(aug_img, 0, 255)
            
        return aug_img
            

    # Random shift
    def shift(self, aug_coordinates, input_img, bg_img):
        x_min = int(min([sublist[1] for sublist in aug_coordinates]))
        y_min = int(min(([sublist[0] for sublist in aug_coordinates])))
        new_coordinates = []
        new_coordinates = [[i[0]-y_min, i[1]-x_min] for i in aug_coordinates]
        
        # Annotated area's maximum height and width
        x_max = int(max([sublist[1] for sublist in new_coordinates]))
        y_max = int(max(([sublist[0] for sublist in new_coordinates])))
        
        if x_max>bg_img.shape[0] or y_max>bg_img.shape[1]:
            return (input_img, aug_coordinates)   

        # Shift the annotated object to origin in original image
        aug_img = np.roll(input_img, -x_min, axis = 0)
        aug_img = np.roll(aug_img, -y_min, axis = 1)
        
        # Random Shift
        x_shift = np.random.randint(0,  bg_img.shape[0] - x_max)
        y_shift = np.random.randint(0, bg_img.shape[1] - y_max)
        aug_coordinates = []
        aug_coordinates = [[i[0]+y_shift, i[1]+x_shift] for i in new_coordinates]       
        aug_img = shape_adjustment().shape_adjust(aug_img, bg_img.shape[1], bg_img.shape[0])            
        aug_img = np.roll(aug_img, x_shift, axis=0)
        aug_img = np.roll(aug_img, y_shift, axis=1)
                 
        return (aug_img, aug_coordinates)
    
    # Grayscale
    def grayscale(self, bg_img, res, gray_choice):
        if(gray_choice!=0 and gray_choice!=1 and gray_choice!=2):
            ch = np.random.randint(0,3)
            gray_choice = ch                                
        if gray_choice == 0:
            bg_img = bg_img[:, :, 0]
            res = res[:, :, 0]
        elif gray_choice == 1:
            rb = res[:, :, 0]
            gb = res[:, :, 0]
            bb = res[:, :, 0]
            res = np.dstack(tup=(rb, gb, bb))
        elif gray_choice == 2:
            rb = bg_img[:, :, 0]
            gb = bg_img[:, :, 0]
            bb = bg_img[:, :, 0]
            bg_img = np.dstack(tup=(rb, gb, bb)) 
        return (bg_img, res)
    
    # Edge Detection
    def edgedetection(self, bg_img, res,edge_choice):
        if(edge_choice!=0 and edge_choice!=1):
            ch = np.random.randint(0,2)
            edge_choice = ch 
 
        if len(res.shape)==2:
            if edge_choice == 0:
                bg_img = np.uint8(bg_img)
                bg_img = cv2.GaussianBlur(bg_img, (3,3), 0)
                bg_img = cv2.Canny(image= bg_img, threshold1=100, threshold2=200)
            res = np.uint8(res)
            res = cv2.GaussianBlur(res, (3,3), 0)
            res = cv2.Canny(image=res, threshold1=100, threshold2=200)
            
        elif len(res.shape)==3:
            if edge_choice == 0:
                bg_img = bg_img[:, :, 0]
                bg_img = np.uint8(bg_img)
                bg_img = cv2.GaussianBlur(bg_img, (3,3), 0)
                bg_img = cv2.Canny(image=bg_img, threshold1=100, threshold2=200)
                bg_img = np.dstack(tup=(bg_img, bg_img, bg_img))
            res = res[:, :, 0]
            res = np.uint8(res)
            res = cv2.GaussianBlur(res, (3,3), 0)
            res = cv2.Canny(image=res, threshold1=100, threshold2=200)
            res = np.dstack(tup=(res, res, res))
        return (bg_img, res)