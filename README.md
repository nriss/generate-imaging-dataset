# DATASET MICROSCOPY #


Python script used to generate patches used by CSBDeep for super resolution microscopy.

Create patches where common spots have been found between multiple image stacks.

Contains modified code of picasso (https://github.com/jungmannlab/picasso) for spot localization.
Also contains modified patches creation function from CSBDeep (https://github.com/CSBDeep/CSBDeep).


Getting started
---------------
To work with DatasetMicroscopy :
1) save your image files (*.tif*) in data_ome_tif  (Equivalent stack images must have the same names)
- **target** for great quality images
- **source** for poor quality images


2) Run generateData.py (python generateData.py)

Example of directory :
1) data/target 
- file1.ome.tif
- file2.ome.tif
- file3.ome.tif
- file4.ome.tif
2) data/source
- file1.ome.tif
- file2.ome.tif
- file3.ome.tif
- file4.ome.tif

The files must have the same names between target and source directories
