## Localised Elastic Deformation
Elastic Deformation augmentation that is localised to a positive segmentation region in a mask. Enables smooth and natural augmentation by enlarging or reducing size of segmentation area without distorting nearby pixels. The amount, radius, direction and randomness of the displacement can be adjusted using the config in `main`, and is applied to both image and mask. <br />

Particularly applicable for medical image segmentation tasks where general Elastic Deformation can cause implausible boundaries. <br />

![alt text](https://github.com/kitbransby/Localised-Elastic-Deformation/blob/main/results/1_plot.png)

## Usage:

Simply run the command below for default config on example data 
```python main.py```


For your own data, place image and masks as npy files into the `data` folder, as follows:
```
Localised-Elastic-Deformation
│
└───data
    │   image_1.npy
    │   image_2.npy
    │   mask_1.npy
    │   mask_2.npy
    │   ...

```


## Dependencies:
Simple ITK (tested on 2.2.0) <br />
numpy <br />
matplotlib <br />
glob <br />

