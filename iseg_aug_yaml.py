import numpy as np
import cv2
import json
import base64
import os
import argparse
from pathlib import Path
import random
from scipy import ndimage
import math
import yaml

class transforms_p():
    
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
        aug_img = shape_adjust(aug_img, bg_img.shape[1], bg_img.shape[0])            
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

class ImageAugmentation(transforms_p):
    
    def __init__(self, yamldata):
        super().__init__()
        self.yamldata = yamldata
        self.aimg_folderpath = self.yamldata['inputs']['aimg_folderpath']
        self.ajson_folderpath = self.yamldata['inputs']['ajson_folderpath']
        self.bgimg_folderpath = self.yamldata['inputs']['bgimg_folderpath']
        self.output_folderpath = self.yamldata['inputs']['output_folderpath']
        self.bg_count = self.yamldata['inputs']['bg_count']
        self.ntimes_perbg = self.yamldata['inputs']['ntimes_perbg']
        self.ratio_threshold = self.yamldata['inputs']['ratio_threshold']
        self.user_class = self.yamldata['inputs']['user_class']

    def pipeline(self):

        # Extracting name of images
        inp_imgs =  os.listdir(self.aimg_folderpath)
        bg_imgs = os.listdir(self.bgimg_folderpath)
        
        ts = self.yamldata['transforms']

        if ts['rotation']['rotation_state'] == True:
            rotlimitangle = ts['rotation']['rotlimitangle']
            if rotlimitangle>=360: 
                rotlimitangle = 359    
                
        flag = 0

        # Iterating on annotated images
        for img in inp_imgs:        

            data, coordinates, shape_type, aug_path, anno_img = dataread(img, self.aimg_folderpath, self.ajson_folderpath, self.output_folderpath, self.user_class)  
            
            length = len(coordinates) 
            
            if length==0:
                print("%s does not contain given class" % (Path(img).stem))
                continue

            height, width = anno_img.shape[0], anno_img.shape[1]            
            counter = 1  # Counts the number of augmentation per annotated image

            # Iterating through a given no. of background images 
            for backgrounds in range(0, self.bg_count):

                bg = os.path.join(self.bgimg_folderpath, random.choice(bg_imgs)) # Selecting a random background image
                bg_img = cv2.imread(bg)                 
                dummy_bg = bg_img.copy()
               
                
                # Checks whether annotated area when pasted in background image is above a threshold or not                 
                list_choice = []
                for i in range(0,len(coordinates)):
                    if checkarea(self, bg_img, coordinates[i], self.ratio_threshold) == True:
                        list_choice.append(i)
                
                if len(list_choice)==0:
                    continue

                # Transforms 
                for j in range(0, self.ntimes_perbg):               
                    
                    rchoice = random.choice(list_choice)
                    aug_coordinates = coordinates[rchoice]
                    aug_anno_img = anno_img
                    
                    # Downscale
                    if ts['scaling']['scaling_state'] == True:
                        downscaleprob = random.random()
                        if ts['scaling']['downscale_prob'] == 1.0 or (downscaleprob <= ts['scaling']['downscale_prob'] and downscaleprob>0):
                            new_coordinates = [[i[0]/ts['scaling']['downscale_factor'], i[1]/ts['scaling']['downscale_factor']] for i in aug_coordinates]
                            if checkarea(self, bg_img, new_coordinates, self.ratio_threshold) == True:
                                aug_anno_img, aug_coordinates = obj.downscale(aug_anno_img, aug_coordinates, ts['scaling']['downscale_factor'])
                                
                    # Rotate
                    if ts['rotation']['rotation_state'] == True:
                        rotationprob = random.random()
                        if ts['rotation']['rotation_prob'] == 1.0 or (rotationprob <= ts['rotation']['rotation_prob'] and rotationprob>0):
                            if rotlimitangle != 0:
                                aug_anno_img, aug_coordinates = obj.rotation(aug_anno_img, aug_coordinates, rotlimitangle)
                    
                    # Upscale
                    if ts['scaling']['scaling_state'] == True:      
                        upscaleprob = random.random()
                        if ts['scaling']['upscale_prob'] == 1.0 or (upscaleprob <= ts['scaling']['upscale_prob'] and upscaleprob>0):
                            new_coordinates = [[i[0]*ts['scaling']['upscale_factor'], i[1]*ts['scaling']['upscale_factor']] for i in aug_coordinates]
                            y_min, y_max, x_min, x_max = findminmax(new_coordinates)
                            if (y_max < bg_img.shape[1] and y_min > 0 and x_max < bg_img.shape[0] and x_min > 0):
                                aug_anno_img, aug_coordinates = obj.upscale(aug_anno_img, aug_coordinates,ts['scaling']['upscale_factor'])
                    

                    # Flip 
                    if ts['flipping']['flipping_state'] == True:        
                        verticalprob =  random.random()
                        if ts['flipping']['vertical_flip_prob'] == 1.0 or (verticalprob <= ts['flipping']['vertical_flip_prob'] and verticalprob>0):                 
                             aug_anno_img, aug_coordinates = obj.flipvertical(aug_anno_img, aug_coordinates)  
                                
                        horizontalprob = random.random()
                        if ts['flipping']['horizontal_flip_prob'] == 1.0 or (horizontalprob <= ts['flipping']['horizontal_flip_prob'] and horizontalprob>0):
                            aug_anno_img, aug_coordinates = obj.fliphorizontal(aug_anno_img, aug_coordinates) 
                    
                    # Brightness and Contrast                   
                    if ts['brightness-contrast']['brightness-contrast_state'] == True: 
                        bcprob = random.random()
                        if ts['brightness-contrast']['brightness-contrast_prob'] == 1.0 or (bcprob <= ts['brightness-contrast']['brightness-contrast_prob'] and bcprob>0):
                            aug_anno_img = cv2.convertScaleAbs(aug_anno_img, alpha=ts['brightness-contrast']['alpha'], beta=ts['brightness-contrast']['beta'])
                            
                    # BLur                   
                    if ts['blur']['blur_state'] == True: 
                        blurprob = random.random()
                        if ts['blur']['blur_prob'] == 1.0 or (blurprob <= ts['blur']['blur_prob'] and blurprob>0):
                            aug_anno_img = obj.blur(aug_anno_img, ts['blur']['blur_choice'])
                        
                    # Noise
                    if ts['noise']['noise_state'] == True:
                        noiseprob = random.random()
                        if ts['noise']['noise_prob'] == 1.0 or (noiseprob <=ts['noise']['noise_prob'] and noiseprob>0):
                            aug_anno_img = obj.noise(aug_anno_img, ts['noise']['noise_choice'], ts['noise']['gauss'], ts['noise']['salt&pepper'], ts['noise']['speckle'])
                    
                    # Shift
                    if ts['randomshift']['shift_state'] == True:  
                        shiftprob = random.random()
                        if ts['randomshift']['shift_prob'] == 1.0 or (shiftprob <= ts['randomshift']['shift_prob'] and shiftprob>0):
                            aug_anno_img, aug_coordinates = obj.shift(aug_coordinates, aug_anno_img, bg_img)
                    
                    aug_anno_img = shape_adjust(aug_anno_img, bg_img.shape[1], bg_img.shape[0])
                    aug_coordinates = cropitup(aug_coordinates,bg_img.shape[1], bg_img.shape[0])
                    
                    if len(aug_coordinates)==0:
                        continue
                        
                    mask = np.zeros((aug_anno_img.shape[0], aug_anno_img.shape[1]), dtype=np.uint8)                   
                    points1 = np.round(np.expand_dims(np.array(aug_coordinates),0)).astype('int32')
                    cv2.fillPoly(mask, points1, 255)
                    cv2.fillPoly(bg_img, points1, 0)
                    res = cv2.bitwise_and(aug_anno_img, aug_anno_img, mask = mask)
                    
                    # Final image  
                    if checkarea(self, bg_img, aug_coordinates,self.ratio_threshold) == True:
                        
                        # Grayscale
                        if ts['grayscale']['grayscale_state'] == True:  
                            grayscaleprob = random.random()
                            if ts['grayscale']['grayscale_prob'] == 1.0 or (grayscaleprob <= ts['grayscale']['grayscale_prob'] and grayscaleprob>0):
                                bg_img, res = obj.grayscale(bg_img, res, ts['grayscale']['grayscale_choice'])
                                
                        aug_img = bg_img + res
                        dataformation(aug_img, aug_path, data, shape_type, rchoice, aug_coordinates, counter, self.user_class)
                        counter+=1
                        flag +=1
                    bg_img = dummy_bg.copy()
                    
        print("%d files formed!" % (flag))

def dataread(img, aimg_folderpath, ajson_folderpath, output_folderpath, user_class):
    image_name = Path(img).stem
    anno_img_path = os.path.join(aimg_folderpath, img)
    anno_img_json = os.path.join(ajson_folderpath, image_name + ".json")

    # Access original coordinates
    f = open(anno_img_json,)
    data = json.load(f)      
    f.close()
    
    coordinates = [[]]
    shape_type = [[]]
    
    if user_class == "default":
        coordinates[0] = data['shapes'][0]['points']
        shape_type[0] =  data['shapes'][0]['shape_type']
        if data['shapes'][0]['shape_type']=="rectangle":
            y_min, y_max, x_min, x_max = findminmax(coordinates[0])
            coordinates[0] = [[y_min,x_min],[y_max,x_min],[y_max,x_max],[y_min,x_max]] 
    else:
        i = 0
        j = 0
        for k in data['shapes']:
                if data['shapes'][i]['label'] == user_class:
                    coordinates.append( data['shapes'][i]['points'])
                    shape_type.append( data['shapes'][i]['shape_type'])
                    
                     # Condition specific for bounding box
                    if data['shapes'][i]['shape_type']=="rectangle":
                        y_min, y_max, x_min, x_max = findminmax(coordinates[j])
                        coordinates[j] = [[y_min,x_min],[y_max,x_min],[y_max,x_max],[y_min,x_max]] 
                    j = j + 1
                i = i + 1

    first_value=data["shapes"][0]
    data["shapes"] = []
    data["shapes"].append(first_value)
        
    aug_path = os.path.join(output_folderpath,image_name + "_aug_")          
    anno_img = cv2.imread(anno_img_path) 

    return (data, coordinates, shape_type, aug_path, anno_img)

def checkarea(self, bg_img, coordinates, ratio_threshold = 0.0):
    
    arr = np.array(coordinates)
    correction = arr[-1][1] * arr[0][0] - arr[-1][0] * arr[0][1]
    main_area = np.dot(arr[::,1][:-1],arr[::,0][1:]) - np.dot(arr[::,0][:-1],arr[::,1][1:])
    area = 0.5 * np.abs(main_area + correction)
    ratio = area /(bg_img.shape[0]*bg_img.shape[1]) * 100

    if ratio >= ratio_threshold:
        return True
    else:
        return False           

# Adds shape_adjust or crops the background image accordingly to match dimensions of the annotated image                 
def shape_adjust(res, bg_width, bg_height):
    
    pad_w_tot = res.shape[1] - bg_width
    pad_h_tot = res.shape[0] - bg_height

    pad_bottom, pad_right = pad_h_tot, pad_w_tot

    if pad_w_tot<=0:
        pad_right = -pad_w_tot

    if pad_h_tot<=0:
        pad_bottom = -pad_h_tot

    # Same dimensions
    if pad_w_tot==0 and pad_h_tot==0:                   
        pass

    # Annotated Image dimensions (height, width) > Background Image dimensions
    elif pad_w_tot>0 and pad_h_tot>0:
        res = res[0:res.shape[0]-pad_bottom, 0:res.shape[1]-pad_right, :]

    # Annotated Image's width >= Background Image's width; Annotated Image's height <= Background Image's height
    elif pad_w_tot>=0 and pad_h_tot<=0:
        rb = np.pad(res[:,:,0],((0,pad_bottom),(0,0)), mode='constant', constant_values=0)
        gb = np.pad(res[:,:,1],((0,pad_bottom),(0,0)), mode='constant', constant_values=0)
        bb = np.pad(res[:,:,2],((0,pad_bottom),(0,0)), mode='constant', constant_values=0)       
        res = np.dstack(tup=(rb, gb, bb))
        res = res[:, 0:res.shape[1]-pad_right, :]

    # Annotated Image's width <= Background Image's width; Annotated Image's height >= Background Image's height
    elif pad_w_tot<=0 and pad_h_tot>=0:
        rb = np.pad(res[:,:,0],((0,0),(0,pad_right)), mode='constant', constant_values=0)
        gb = np.pad(res[:,:,1],((0,0),(0,pad_right)), mode='constant', constant_values=0)
        bb = np.pad(res[:,:,2],((0,0),(0,pad_right)), mode='constant', constant_values=0)
        res = np.dstack(tup=(rb, gb, bb))
        res = res[0:res.shape[0]-pad_bottom, :, :]

    # Annotated Image dimensions (height, width) < Background Image dimensions
    else:
        rb = np.pad(res[:,:,0],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        gb = np.pad(res[:,:,1],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        bb = np.pad(res[:,:,2],((0,pad_bottom),(0,pad_right)), mode='constant', constant_values=0)
        res = np.dstack(tup=(rb, gb, bb)) 
   
    return res

def findminmax(coordinates):
    arr = np.array(coordinates)
    y_min = min(arr[::,0])
    y_max = max(arr[::,0])
    x_min = min(arr[::,1])
    x_max = max(arr[::,1])

    return y_min, y_max, x_min, x_max

# Remove the coordinates values exceeding the image

def cropitup(coordinates, width, height):
    
    dummy = []
    index = []
    for i in range(0,len(coordinates)): 
        if coordinates[i][0]<=width and coordinates[i][1]<=height:
            index.append(i)
            
    if len(index) == 0:
        return dummy
    elif len(index) == len(coordinates):
        return coordinates
    else:
        return dummy
    
def dataformation(aug_img, aug_path, data, shape_type, rchoice, new_coordinates1, counter, user_class):
    
    new_path = aug_path + str(counter) + ".jpg"
    cv2.imwrite(new_path, aug_img.astype(np.uint8))

    json_path = aug_path + str(counter) + ".json"

    # Condition specific for bounding box
    if shape_type[rchoice]=="rectangle":    
        ymin,ymax,xmin,xmax = findminmax(new_coordinates1)        
        new_coordinates1 = [[ymin, xmin], [ymax, xmax]] 
     
    if user_class!="default":
        data["shapes"][0]["label"] = user_class
    data["shapes"][0]["shape_type"] = shape_type[rchoice]
    data["shapes"][0]["points"] = new_coordinates1
    data["imagePath"] = ".." + os.path.basename(new_path)
    data["imageData"] = str(base64.b64encode(open(new_path,'rb').read()))[2:-1]
    data["imageHeight"] = aug_img.shape[0]
    data["imageWidth"] = aug_img.shape[1]

    with open(json_path, 'w') as outfile:
        json.dump(data, outfile, indent=2)
        
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", help="Path of YAML file", type=str)
    args = parser.parse_args()
    if os.path.exists(args.yaml_path):
        stream = open(args.yaml_path, 'r')
        try:
            yamldata = yaml.safe_load(stream)
        except:
            print("Not a YAML File!")    
        try:
            obj = ImageAugmentation(yamldata)
            
        except:
            print("YAML File does not contain expected data.")
        obj.pipeline()
    else:
        print("Path does not exist")