 # Image Augmentation
 
This script is used to augment image data created using LabelMe-MIT. It crops the annotated part from the reference image(input annotated image) and paste it randomly to any possible area of the provided background images. 

- Inputs: 
    1. aimg_folderpath - path of annotated image folder
    2. ajson_folderpath - path of annotated image's json folder 
    3. bgimg_folderpath - path of folder containing background images
    4. output_folderpath - path of folder where new augmented files are saved
    5. bg_count - no. of random background images to be used from background image folder
    6. ntimes_perbg - no. of times you wish to augment using a single background image

- Output: 
    Newly augmented images and it's json

- Constraints: 
    1. LabelMe should be used to create the annotated json and thus images should be in .jpg format
    2. Annotation should be a closed polygon/bounding box  
    3. There should be one annotation in an image                
        
## Demo


### Input Image

<img align="left" width="250px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/1.png" />
<img align="left" width="250px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/2.png" />

<br />
<br />
<br />
<br />
<br />
<br />

### Background Images

<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b1.jpg" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b3.jpg" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b4.jpg" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/raw/main/background_images/b5.jpg" />

<br />
<br />
<br />
<br />
<br />
<br />

### Augmented Images

<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/O_1.png" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/O_2.png" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/O_3.png" />
<img align="left" width="180px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/O_4.png" />



