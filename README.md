 # Image Aumentation
 
This script is used to augment image data created using LabelMe-MIT. It crops the annotated part from the reference image(input annotated image) and paste it randomly to any possible area of the provided background images. 

- Inputs: 
    1. anno_img_path - path of annotated image
    2. anno_img_json - path of annotated image's json 
    3. bg_img_folder_path - path of folder containing background images
    4. output_folder - path of folder where new augmented files are saved
    5. ntimes - no. of times you wish to augment using a single background image

- Output: 
    Newly augmented images and it's json

- Constraints: 
    1. Labelme should be used to create the annotated json.
    2. Annotation should be closed polygon               
        
## Demo


### Input Image

<img align="centre" width="500px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/Screenshot_0.png" />

### Background Images

<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b1.jpg" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b3.jpg" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b4.jpg" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b5.jpg" />

### Augmented Images

<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/Screenshot_1.png" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/Screenshot_3.png" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/Screenshot_4.png" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/Screenshot_5.png" />



