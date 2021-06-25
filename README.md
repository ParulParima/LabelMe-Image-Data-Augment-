# Image Data Augmentation

## Introduction

This script is used to augment image data created using **LabelMe-MIT**. It copies the annotated part from the reference image (input annotated image) and paste it randomly to any possible area of the provided background images.

 **It also creates new json files for the newly created images, i.e., it augments both the image as well as it's annotation**.

### Constraints

- LabelMe should be used to create the annotated json and thus images should be in .jpg format
- Annotation should be a closed polygon/bounding box  
- There should be one annotation in an image

## Description

1. Random Shift - It copies the annotated part from the reference image (input annotated image) and paste it randomly to any possible area of the provided background images. 

- **iseg_aug.py** - The new image formed is of same shape as of input annotated image. (**Output 1**)

- **iseg_aug_2.py** - The new image formed is of same shape as of background image. (**Output 2**)
  
2. Rotation - It copies the annotated part from the reference image (input annotated image) and rotates it randomly.Then the copied part is pasted to any possible area of the provided background images.

- **iseg_aug_rotate.py** - The rotation angle is fixed for a specific background but random shift happens in each image. (**Output 3**)
 
- **iseg_aug_random_rotate.py** - The random rotation and random shift happens simultaneously in each and every background image. (**Output 4**)

- **iseg_aug_5.py** - The random rotation and random shift happens simultaneously in each and every background image to a specific user class. *This script is applicable for images with multiple annotations, wherein the user can choose a particular class*.

<table>

<tr>
<th>&nbsp;</th>
<th>Lantern</th>
<th>Apple</th>
<th>Astronaut</th>
</tr>

<!-- Line 1: Inputs -->
<tr>
<td><em>Inputs</em></td>
<td><img align="left" width="250px"  height = "141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/1.png?raw=true" /></td>
<td><img align="left" width="250px" height = "141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Images/2.png?raw=true" /></td>
<td><img align="left" width="250px" height = "141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Images/3.png?raw=true"/></td>
</tr>

<!-- Line 2: Backgrounds -->
<tr>
<td><em>Backgrounds</em></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/background_images/b1.jpg?raw=true" /></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/background_images/b2.jpg?raw=true" /></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/background_images/b3.jpg?raw=true" /></td>
</tr>

<!-- Line 3: Output 1 -->
<tr>
<td><em>Output 1</em></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Bg_Img_Lantern_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Bg_Img_Apple_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Bg_Img_Astronaut_33.gif?raw=true"/></td>
</tr>

<!-- Line 4: Output 2 -->
<tr>
<td><em>Output 2</em></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Anno_Img_Lantern_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Anno_Img_Apple_33.gif?raw=true"/></td>
<td><img align="centre" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Anno_Img_Astronaut_33.gif?raw=true"/></td>
</tr>

<!-- Line 5: Output 3 -->
<tr>
<td><em>Output 3</em></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Rotation_Lantern_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Rotation_Apple_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Rotation_Astronaut_33.gif?raw=true"/></td>
</tr>

<!-- Line 6: Output 4 -->
<tr>
<td><em>Output 4</em></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Random_Rotation_Lantern_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Random_Rotation_Apple_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/master/Gifs/Random_Rotation_Astronaut_33.gif?raw=true"/></td>
</tr>

</table>


3. It copies the annotated part from the reference image(input annotated image) and do different tranforms on it like random rotation, flip, upscale, downscale, random blur, random noise and random shift. All these tranformations will happen as per the user requirement which can be specified on the YAML file. Then the transformed annotation is pasted to any provided random background images.
*This script is applicable for images with multiple annotations, wherein the user can choose a particular class. If not choosen, it will consider the first annotation in the annotated image*.  
- **iseg_aug_yaml.py** 
- **input.yaml** 
   

<table>

<tr>
<th>&nbsp;</th>
<th>Lantern</th>
<th>Apple</th>
<th>Astronaut</th>
</tr>
<!-- Line 1: Output -->
<tr>
<td><em>Output</em></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Gifs/YLantern_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Gifs/YApple_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Gifs/YAstronaut_33.gif?raw=true"/></td>
</tr>

</table>