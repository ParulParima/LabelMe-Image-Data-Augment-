# Image Data Augmentation

## Introduction

This script is used to augment image data created using **LabelMe-MIT**.

 **It also creates new json files for the newly generated images, i.e., it augments both the image as well as it's annotation**.

### Constraints

- LabelMe should be used for annotating images
- Annotation should be a closed polygon/bounding box  
- There should be one annotation in an image

*This script can work for images with multiple annotation as well, but it will only take into account the first annotation or the first user-specified class annotation*.

## Description

It copies the annotated portion from the reference image(input annotated image), processes it according to the user's instructions, which can be provided in the YAML file. And then paste it on one of the given random background images.

- **iseg_aug_yaml.py**
- **input.yaml**

### Transforms

1. Downscale

2. Upscale

3. Rotation

4. Horizontal Flip

5. Vertical Flip

6. Random Shift

7. Blur - Averaging, Gaussian Blurring, Median Blurring, Bilateral Filtering

8. Noise - Gauss, Salt and Pepper, Poisson, Speckle

9. Grayscale

10. Brightness and Contrast

11. Canny Edge Detection

### Options other than transforms

1. Threshold Ratio - Ratio of annotated area to background image area. Combinations below this ratio will be neglected.

2. User Class - Choose a specific class on which to do transformations

3. Pad Annotation - Amount of padding you want to add to the annotation

</br>
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
<td><em>Output</em></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Gifs/YN_Lantern_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Gifs/YN_Apple_33.gif?raw=true"/></td>
<td><img align="left" width="250px" height="141px" src="https://github.com/ParulParima/LabelMe-Image-Data-Augment-/blob/main/Gifs/YN_Astronaut_33.gif?raw=true"/></td>
</tr>
</table>
