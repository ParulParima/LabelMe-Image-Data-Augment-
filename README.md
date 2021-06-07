# LabelMe - Image Data Augmentation

## Introduction

This script is used to augment image data created using LabelMe-MIT. It copies the annotated part from the reference image(input annotated image) and paste it randomly to any possible area of the provided background images. 

### Constraints 

- LabelMe should be used to create the annotated json and thus images should be in .jpg format
- Annotation should be a closed polygon/bounding box  
- There should be one annotation in an image

## Description

1.Random Shift - It copies the annotated part from the reference image(input annotated image) and paste it randomly to any possible area of the provided background images. 

- iseg_aug.py - The new image formed is of same shape as of input annotated image.  

- iseg_aug_2.py - The new image formed is of same shape as of background image.
  
2.Rotation - It copies the annotated part from the reference image(input annotated image) and rotates it randomly.Then the copied part is pasted to any possible area of the provided background images.

- iseg_aug_rotate.py - The rotation angle is fixed for a specific background but random shift happens in each image.
 
- iseg_aug_random_rotate.py - The random rotation and random shift happens simultaneously in each and every background image.

### Input Image

<br />

<tr>
<td><img align="left" width="250px"  height = "141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/1.png?raw=true" /><img  align="left" width="250px" height = "141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/2.png?raw=true" /><img align="left" width="250px" height = "141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Images/3.png?raw=true"/></td>
</tr>

<br />
<br />
<br />
<br />
<br />

### Background Images

<br />

<tr>
<td><img align="left" width="200px" height="112px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/background_images/b1.jpg?raw=true" /><img align="left" width="200px" height="112px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/background_images/b2.jpg?raw=true" /><img align="left" width="200px" height="112px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/background_images/b3.jpg?raw=true" /><img align="left" width="200px" height="112px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/background_images/b4.jpg?raw=true" /></td>
</tr>

<br />
<br />
<br />
<br />
<br />

### Augmented Images for iseg_aug.py

<br />

<tr>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Bg_Img_Lantern_33.gif?raw=true"/><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Bg_Img_Apple_33.gif?raw=true"/><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Bg_Img_Astronaut_33.gif?raw=true"/>
</td>
</tr>

<br />
<br />
<br />
<br />
<br />

### Augmented Images - iseg_aug_2.py

<br />

<tr>
<td><img align="left" width="250px" height="141px"src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Anno_Img_Lantern_33.gif?raw=true"/><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Anno_Img_Apple_33.gif?raw=true"/><img align="left" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Anno_Img_Astronaut_33.gif?raw=true"/>
</td>
</tr>

<br />
<br />
<br />
<br />
<br />

### Augmented Images - iseg_aug_rotate.py

<br />

<tr>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Rotation_Lantern_33.gif?raw=true"/><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Rotation_Apple_33.gif?raw=true"/><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Rotation_Astronaut_33.gif?raw=true"/>
</td>
</tr>

<br />
<br />
<br />
<br />
<br />

### Augmented Images - iseg_aug_random_rotate.py

<br />

<tr>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Random_Rotation_Lantern_33.gif?raw=true"/><img width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Random_Rotation_Apple_33.gif?raw=true"/><img width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/Branch1/Gifs/Random_Rotation_Astronaut_33.gif?raw=true"/>
</td>
</tr>