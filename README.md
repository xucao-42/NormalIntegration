#[Normal Integration via Inverse Plane Fitting with Minimum Point-to-Plane Distance](https://openaccess.thecvf.com/content/CVPR2021/html/Cao_Normal_Integration_via_Inverse_Plane_Fitting_With_Minimum_Point-to-Plane_Distance_CVPR_2021_paper.html)
[Xu Cao](https://hoshino042.github.io/homepage/), [Boxin Shi](http://alumni.media.mit.edu/~shiboxin/), [Fumio Okura](http://alumni.media.mit.edu/~shiboxin/) and [Yasuyuki Matsushita](http://www-infobiz.ist.osaka-u.ac.jp/en/member/matsushita/)

CVPR 2021

![](teaser.png)

This repository contains the official python implementation of the CVPR'21 normal integration paper, along with our python implementations based on the following papers:
- "Variational Methods for Normal Integration", Quéau et al., Journal of Mathematical Imaging and Vision 60(4), pp 609--632, 2018. 
- "Normal Integration: a Survey", Quéau et al., Journal of Mathematical Imaging and Vision 60(4), pp 576--593, 2018. [Official Matlab Code](https://github.com/yqueau/normal_integration)
- "Least Squares Surface Reconstruction on Arbitrary Domains", Zhu et al., ECCV, 2020. [Official Matlab Code](https://github.com/waps101/LSQSurfaceReconstruction)
- "Surface-from-Gradients: An Approach Based on Discrete Geometry Processing", Xie et al., CVPR, 2014.
# Quick Start 
 - cd to this repository's root folder and reproduce our anaconda environment by running
 
 ```
 conda env create -f=environment.yml 
 conda activate ni
 ```
 
 - run the demo code
 
 ```python comparison_on_analytically_computed_orthographic_normal_maps.py```
 
 This script compares 5 methods on 3 orthographic normal maps: sphere, vase, and anisotropic Gaussian.
 The results will be saved in `results/#TIME`.
 
 You can optionally add Gaussion noise and/or outliers to the input normal maps by running

  ```
  python comparison_on_analytically_computed_orthographic_normal_maps.py --noise 0.1
  python comparison_on_analytically_computed_orthographic_normal_maps.py --outlier 0.1
  python comparison_on_analytically_computed_orthographic_normal_maps.py --outlier 0.1 --noise 0.1
  ```
  The number after `--noise` is the standard deviation of Gaussian noise added to all normal vectors; the number after `--outlier` is the percentage (0~1) of outliers in the normal map.

- To visualize the estimated mesh surfaces, run

``` python plot_surface.py --path #YOUR_FOLDER_CONTAINING_PLY_FILES```

A plot window of one surface will pop up, you can adjust the viewpoint that you would like to save as images.
Then close the window, the images of all meshes viewed from the selected viewpoint will be saved in your input folder. 

# Use Your Data

Under construction.
