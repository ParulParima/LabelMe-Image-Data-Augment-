import numpy as np
import math
import random
import cv2
import json
import base64
import os
import argparse
from pathlib import Path
import yaml
from transforms import *
from shape_adjustment import *

class ImageAugmentation:
    
    def __init__(self, yamldata):
        self.yamldata = yamldata
        self.aimg_folderpath = self.yamldata['inputs']['aimg_folderpath']
        self.ajson_folderpath = self.yamldata['inputs']['ajson_folderpath']
        self.bgimg_folderpath = self.yamldata['inputs']['bgimg_folderpath']
        self.output_folderpath = self.yamldata['inputs']['output_folderpath']
        self.bg_count = self.yamldata['inputs']['bg_count']
        self.ntimes_perbg = self.yamldata['inputs']['ntimes_perbg']
        self.ratio_threshold = self.yamldata['inputs']['ratio_threshold']
        self.user_class = self.yamldata['inputs']['user_class']
        self.pad_annotation = self.yamldata['inputs']['pad_annotation']

    def dataread(self,img, aimg_folderpath, ajson_folderpath, output_folderpath, user_class, pad):
        image_name = Path(img).stem
        anno_img_path = os.path.join(aimg_folderpath, img)
        anno_img_json = os.path.join(ajson_folderpath, image_name + ".json")
        aug_path = os.path.join(output_folderpath,image_name + "_aug_")          
        anno_img = cv2.imread(anno_img_path)
        height, width = anno_img.shape[0], anno_img.shape[1]
        
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
                x_min = x_min - pad if x_min - pad>=0 else 0
                y_min = y_min - pad if y_min - pad>=0 else 0
                x_max = x_max + pad if x_max + pad<=height else height
                y_max = y_max + pad if y_max + pad<=width else width
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
                            x_min = x_min - pad if x_min - pad>=0 else 0
                            y_min = y_min - pad if y_min - pad>=0 else 0
                            x_max = x_max + pad if x_max + pad<=height else height
                            y_max = y_max + pad if y_max + pad<=width else width
                            coordinates[j] = [[y_min,x_min],[y_max,x_min],[y_max,x_max],[y_min,x_max]] 
                        j = j + 1
                    i = i + 1

        first_value=data["shapes"][0]
        data["shapes"] = []
        data["shapes"].append(first_value)    

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

    def findminmax(self,coordinates):
        arr = np.array(coordinates)
        y_min = min(arr[::,0])
        y_max = max(arr[::,0])
        x_min = min(arr[::,1])
        x_max = max(arr[::,1])

        return y_min, y_max, x_min, x_max

    # Remove the coordinates values exceeding the image
    def cropitup(self,coordinates, width, height):    
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

    def dataformation(self,aug_img, aug_path, data, shape_type, rchoice, new_coordinates1, counter, user_class):    
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

            data, coordinates, shape_type, aug_path, anno_img = obj.dataread(img, self.aimg_folderpath, self.ajson_folderpath, self.output_folderpath, self.user_class, self.pad_annotation)  
            
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
                    if obj.checkarea(bg_img, coordinates[i], self.ratio_threshold) == True:
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
                            if obj.checkarea(bg_img, new_coordinates, self.ratio_threshold) == True:
                                aug_anno_img, aug_coordinates = transforms().downscale(aug_anno_img, aug_coordinates, ts['scaling']['downscale_factor'])
                                
                    # Rotate
                    if ts['rotation']['rotation_state'] == True:
                        rotationprob = random.random()
                        if ts['rotation']['rotation_prob'] == 1.0 or (rotationprob <= ts['rotation']['rotation_prob'] and rotationprob>0):
                            if rotlimitangle != 0:
                                aug_anno_img, aug_coordinates = transforms().rotation(aug_anno_img, aug_coordinates, rotlimitangle)
                    
                    # Upscale
                    if ts['scaling']['scaling_state'] == True:      
                        upscaleprob = random.random()
                        if ts['scaling']['upscale_prob'] == 1.0 or (upscaleprob <= ts['scaling']['upscale_prob'] and upscaleprob>0):
                            new_coordinates = [[i[0]*ts['scaling']['upscale_factor'], i[1]*ts['scaling']['upscale_factor']] for i in aug_coordinates]
                            y_min, y_max, x_min, x_max = obj.findminmax(new_coordinates)
                            if (y_max < bg_img.shape[1] and y_min > 0 and x_max < bg_img.shape[0] and x_min > 0):
                                aug_anno_img, aug_coordinates = transforms().upscale(aug_anno_img, aug_coordinates,ts['scaling']['upscale_factor'])
                    

                    # Flip 
                    if ts['flipping']['flipping_state'] == True:        
                        verticalprob =  random.random()
                        if ts['flipping']['vertical_flip_prob'] == 1.0 or (verticalprob <= ts['flipping']['vertical_flip_prob'] and verticalprob>0):                 
                             aug_anno_img, aug_coordinates = transforms().flipvertical(aug_anno_img, aug_coordinates)  
                                
                        horizontalprob = random.random()
                        if ts['flipping']['horizontal_flip_prob'] == 1.0 or (horizontalprob <= ts['flipping']['horizontal_flip_prob'] and horizontalprob>0):
                            aug_anno_img, aug_coordinates = transforms().fliphorizontal(aug_anno_img, aug_coordinates) 
                    
                    # Brightness and Contrast                   
                    if ts['brightness-contrast']['brightness-contrast_state'] == True: 
                        bcprob = random.random()
                        if ts['brightness-contrast']['brightness-contrast_prob'] == 1.0 or (bcprob <= ts['brightness-contrast']['brightness-contrast_prob'] and bcprob>0):
                            aug_anno_img = cv2.convertScaleAbs(aug_anno_img, alpha=ts['brightness-contrast']['alpha'], beta=ts['brightness-contrast']['beta'])
                            
                    # BLur                   
                    if ts['blur']['blur_state'] == True: 
                        blurprob = random.random()
                        if ts['blur']['blur_prob'] == 1.0 or (blurprob <= ts['blur']['blur_prob'] and blurprob>0):
                            aug_anno_img = transforms().blur(aug_anno_img, ts['blur']['blur_choice'])
                        
                    # Noise
                    if ts['noise']['noise_state'] == True:
                        noiseprob = random.random()
                        if ts['noise']['noise_prob'] == 1.0 or (noiseprob <=ts['noise']['noise_prob'] and noiseprob>0):
                            aug_anno_img = transforms().noise(aug_anno_img, ts['noise']['noise_choice'], ts['noise']['gauss'], ts['noise']['salt&pepper'], ts['noise']['speckle'])
                    
                    # Shift
                    if ts['randomshift']['shift_state'] == True:  
                        shiftprob = random.random()
                        if ts['randomshift']['shift_prob'] == 1.0 or (shiftprob <= ts['randomshift']['shift_prob'] and shiftprob>0):
                            aug_anno_img, aug_coordinates = transforms().shift(aug_coordinates, aug_anno_img, bg_img)
                    
                    aug_anno_img = shape_adjustment().shape_adjust(aug_anno_img, bg_img.shape[1], bg_img.shape[0])
                    aug_coordinates = obj.cropitup(aug_coordinates,bg_img.shape[1], bg_img.shape[0])
                    
                    if len(aug_coordinates)==0:
                        continue
                        
                    mask = np.zeros((aug_anno_img.shape[0], aug_anno_img.shape[1]), dtype=np.uint8)                   
                    points1 = np.round(np.expand_dims(np.array(aug_coordinates),0)).astype('int32')
                    cv2.fillPoly(mask, points1, 255)
                    cv2.fillPoly(bg_img, points1, 0)
                    res = cv2.bitwise_and(aug_anno_img, aug_anno_img, mask = mask)
                    
                    # Final image  
                    if obj.checkarea(bg_img, aug_coordinates,self.ratio_threshold) == True:
                        
                        # Grayscale
                        if ts['grayscale']['grayscale_state'] == True:  
                            grayscaleprob = random.random()
                            if ts['grayscale']['grayscale_prob'] == 1.0 or (grayscaleprob <= ts['grayscale']['grayscale_prob'] and grayscaleprob>0):
                                bg_img, res = transforms().grayscale(bg_img, res, ts['grayscale']['grayscale_choice'])
                                
                        # Edge Detection    
                        if ts['edgedetection']['edgedetection_state'] == True:  
                            edgedetectionprob = random.random()
                            if ts['edgedetection']['edgedetection_prob'] == 1.0 or (edgedetectionprob <= ts['edgedetection']['edgedetection_prob'] and edgedetectionprob>0):
                                bg_img, res = transforms().edgedetection(bg_img, res, ts['edgedetection']['edgedetection_choice'])
                                
                        aug_img = bg_img + res
                        obj.dataformation(aug_img, aug_path, data, shape_type, rchoice, aug_coordinates, counter, self.user_class)
                        counter+=1
                        flag +=1
                    bg_img = dummy_bg.copy()
                    
        print("%d files formed!" % (flag))
        
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