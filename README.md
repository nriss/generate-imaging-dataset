# DatasetMicroscopy

######################
# DATASET MICROSCOPY #
######################

Python script used to generate patches used by CSBDeep for super resolution microscopy.

Create patches where common points have been found between multiple image stacks.

Contains modified code of picasso (https://github.com/jungmannlab/picasso) for spot localization.
Also contains modified patches creation function from CSBDeep (https://github.com/CSBDeep/CSBDeep).

To work with DatasetMicroscopy :
- save your image files (*.tif*) in data_ome_tif 
--> net for great quality images
--> flou for bad quality images
The equivalent images must have the same names.

Then run generateData.py (python generateData.py)
