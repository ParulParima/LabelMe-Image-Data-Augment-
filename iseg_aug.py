import numpy as np
import cv2
import json
import base64
import glob

"""
This function, iseg_aug, is used to augment image data.

    Inputs: 1. anno_img_path - path of annotated image
            2. anno_img_json - path of annotated image's json 
            3. bg_img_folder_path - path of folder containing background images
            4. output_folder - path of folder where new augmented files are saved
            5. ntimes - no. of times you wish to augment using a single background image
            
    Output: Newly augmented images and it's json
            
    Constraints: 1. Labelme should be used to create the annotated json.
                 2. Annotation should be closed polygon                  
"""

def iseg_aug(anno_img_path,anno_img_json,bg_img_folder_path,output_folder,ntimes):
    
    # Access original coordinates
    f = open(anno_img_json,)
    data = json.load(f)      
    coordinates = [i for i in data['shapes'][0]['points']]  
    f.close()
    
    if data['shapes'][0]['shape_type']=="rectangle":
        ymax = max(coordinates[0][0],coordinates[1][0])
        ymin = min(coordinates[0][0],coordinates[1][0])
        xmax = max(coordinates[0][1],coordinates[1][1])
        xmin = min(coordinates[0][1],coordinates[1][1])
        coordinates = [[ymin,xmin],[ymax,xmin],[ymax,xmax],[ymin,xmax]]
        
    aug_path = output_folder + "\\" + anno_img_path.split('\\')[-1].split('.')[0] + "_aug" 
    
    anno_img = cv2.imread(anno_img_path)    
    height, width = anno_img.shape[0],anno_img.shape[1]
    
    counter=0                                             # Counts the no.of background images
    
    for bg in glob.glob(r"%s\*.jpg"%(bg_img_folder_path)):
    
        bg_img = cv2.imread(bg)
       
        # Adds padding or crops the background image accordingly to match dimensions of the annotated image   
        pad_w_tot = bg_img.shape[1] - anno_img.shape[1]
        pad_h_tot = bg_img.shape[0] - anno_img.shape[0]
               
        pad_top, pad_bottom, pad_left, pad_right =0, 0, 0, 0
        
        if pad_w_tot%2==0:
            if pad_w_tot<=0:
                pad_left, pad_right = int(-pad_w_tot/2), int(-pad_w_tot/2)
            else:
                pad_left, pad_right = int(pad_w_tot/2), int(pad_w_tot/2)
        else:
            if pad_w_tot<=0:
                pad_left, pad_right = int(-pad_w_tot/2), int(-pad_w_tot/2) + 1
            else:
                pad_left, pad_right = int(pad_w_tot/2), int(pad_w_tot/2) + 1
            
        if pad_h_tot%2==0:
            if pad_h_tot<=0:
                pad_top, pad_bottom = int(-pad_h_tot/2), int(-pad_h_tot/2)
            else:
                pad_top, pad_bottom = int(pad_h_tot/2), int(pad_h_tot/2)
        else:
            if pad_h_tot<=0:
                pad_top, pad_bottom = int(-pad_h_tot/2), int(-pad_h_tot/2) + 1
            else:
                pad_top, pad_bottom = int(pad_h_tot/2), int(pad_h_tot/2) + 1
                
        if pad_w_tot==0 and pad_h_tot==0:
            pass
        elif pad_w_tot>0 and pad_h_tot>0:
            bg_img = bg_img[pad_top:bg_img.shape[0]-pad_bottom, pad_left:bg_img.shape[1]-pad_right, :]
            
        elif pad_w_tot>=0 and pad_h_tot<=0:
            rb = np.pad(bg_img[:,:,0],((pad_top,pad_bottom),(0,0)), mode='constant', constant_values=255)
            gb = np.pad(bg_img[:,:,1],((pad_top,pad_bottom),(0,0)), mode='constant', constant_values=255)
            bb = np.pad(bg_img[:,:,2],((pad_top,pad_bottom),(0,0)), mode='constant', constant_values=255)       
            bg_img = np.dstack(tup=(rb, gb, bb))
            bg_img = bg_img[:, pad_left:bg_img.shape[1]-pad_right, :]

        elif pad_w_tot<=0 and pad_h_tot>=0:
            rb = np.pad(bg_img[:,:,0],((0,0),(pad_left,pad_right)), mode='constant', constant_values=255)
            gb = np.pad(bg_img[:,:,1],((0,0),(pad_left,pad_right)), mode='constant', constant_values=255)
            bb = np.pad(bg_img[:,:,2],((0,0),(pad_left,pad_right)), mode='constant', constant_values=255)
            bg_img = np.dstack(tup=(rb, gb, bb))
            bg_img = bg_img[pad_top:bg_img.shape[0]-pad_bottom, :, :]

        else:
            rb = np.pad(bg_img[:,:,0],((pad_top,pad_bottom),(pad_left,pad_right)), mode='constant', constant_values=255)
            gb = np.pad(bg_img[:,:,1],((pad_top,pad_bottom),(pad_left,pad_right)), mode='constant', constant_values=255)
            bb = np.pad(bg_img[:,:,2],((pad_top,pad_bottom),(pad_left,pad_right)), mode='constant', constant_values=255)
            bg_img = np.dstack(tup=(rb, gb, bb))

        # Shifts the annotated part of image to origin to allow random movement in background image
        dummy_bg = bg_img.copy()   
        x_min = int(min([sublist[1] for sublist in coordinates]))
        y_min = int(min(([sublist[0] for sublist in coordinates])))

        new_coordinates = [[i[0]-y_min,i[1]-x_min] for i in coordinates]

        x_max = int(max([sublist[1] for sublist in new_coordinates]))
        y_max = int(max(([sublist[0] for sublist in new_coordinates])))
        
        # Forms mask
        mask = np.zeros((height, width), dtype=np.uint8)
        points1 = np.round(np.expand_dims(np.array(coordinates),0)).astype('int32')
        cv2.fillPoly(mask, points1,255)
        res = cv2.bitwise_and(anno_img,anno_img,mask = mask)
        dummy_res = res.copy()

        random_moves = []
        
        # Calculates the new random coordinates for the annotated object in the background image as well as saves the newly formed image and it's json
        for j in range(0,ntimes):


            x_shift = np.random.randint(0,  height - x_max)
            y_shift = np.random.randint(0, width - y_max)

            if (len(random_moves)-1)== ((height - x_max+1)*(width - y_max+1)):
                break

            while 1==1:
                if [x_shift,y_shift] not in random_moves:
                    random_moves.append([x_shift,y_shift])
                    break
                else:
                    x_shift = np.random.randint(0,  height - x_max)
                    y_shift = np.random.randint(0, width - y_max)

            new_coordinates1 = [[i[0]+y_shift,i[1]+x_shift] for i in new_coordinates]

            points2 = np.round(np.expand_dims(np.array(new_coordinates1),0)).astype('int32')   
            bg_img = dummy_bg.copy()
            cv2.fillPoly(bg_img, points2,0)
            res = dummy_res.copy()
            res = np.roll(res,x_shift-x_min,axis=0)
            res = np.roll(res, y_shift-y_min ,axis=1)
            aug_img = res + bg_img

            n = str(counter) + "_" + str(j)
            new_path = aug_path + n + ".jpg"
            cv2.imwrite(new_path, aug_img.astype(np.uint8))

            json_path = aug_path + n + ".json"
            
            if data['shapes'][0]['shape_type']=="rectangle":
                new_coordinates1 = [new_coordinates1[0],new_coordinates1[2]]                
            data["shapes"][0]["points"] = new_coordinates1  
            data["imagePath"] = "..\\" + new_path.split('\\')[-1]
            data["imageData"] = str(base64.b64encode(open(new_path,'rb').read()))[2:-1]

            with open(json_path, 'w') as outfile:
                json.dump(data, outfile, indent=2)

        counter+=1