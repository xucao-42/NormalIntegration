The official code of CVPR'21 paper "Normal Integration via Inverse Plane Fitting with Minimum Point-to-Plane Distance".

# Quick Start 
 - create an anaconda environment with the required dependencies by running
 
 ```
 conda create --name ni --file requirements.txt
 conda activate ni
 ```
 
 - run the demo code
 
 ```python comparison_on_analytically_computed_orthographic_normal_maps.py```
 
 This scipt compares 5 methods on 3 orthographic normal maps: sphere, vase, and anistropic Gaussian.
 The results will be saved in `results/#TIME`.
 
 You can optionally add Gaussion noise and/or outliers to the input normal maps by running

  ```
  python comparison_on_analytically_computed_orthographic_normal_maps.py --noise 0.1
  python comparison_on_analytically_computed_orthographic_normal_maps.py --outlier 0.1
  python comparison_on_analytically_computed_orthographic_normal_maps.py --outlier 0.1 --noise 0.1
  ```
  The number after `--noise` is the standard deviation of Guassian noise added to all normal vectors; the number after `--outlier` is the percentage (0~1) of outliers in the normal map.

- To visualize the estimated mesh surfaces, run

``` python plot_surface.py --path #YOUR_FOLDER_CONTAINING_PLY_FILES```

A plot window of one surface will pop up, you can adjust the viewpoint that you would like to save as images.
Then close the window, the images of all meshes viewed from the selected viewpoint will be saved in your input folder. 

# Use Your Data

Under construction.
