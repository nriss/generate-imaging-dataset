# DATASET MICROSCOPY #


Python script used to generate patches used by CSBDeep for super resolution microscopy.

Create patches where common spots have been found between multiple image stacks.

Contains modified code of picasso (https://github.com/jungmannlab/picasso) for spot localization.
Also contains modified patches creation function from CSBDeep (https://github.com/CSBDeep/CSBDeep).


Getting started
---------------
To work with DatasetMicroscopy :
1) save your image files (*.tif*) in data_ome_tif  (Equivalent stack images must have the same names)
- **net** for great quality images
- **flou** for bad quality images


2) Run generateData.py (python generateData.py)
