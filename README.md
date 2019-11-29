# DATASET MICROSCOPY #


Python script used to generate patches and create CSBDeep model applied to super resolution microscopy.
1) Create patches where common spots have been found between image stack pairs.
2) Build CSBDeep model
3) Test CSBDeep model


Getting started
---------------
To work with DatasetMicroscopy :
1) Save your image files (*.tif*) in a data directory
- **target** for great quality images
- **source** for poor quality images


Example of directory
- data/target/{file1.ome.tif, file2.ome.tif, file3.ome.tif, file4.ome.tif}
- data/source/{file1.ome.tif, file2.ome.tif, file3.ome.tif, file4.ome.tif}

The equivalent image stacks must have the same names between target and source directories


2) Run generateData.py (python generateData.py)

This script will create a npz file containing pair of image patch needed to build the model

3) Run createModel.py (python createModel.py)

This function will use the npz file and CSBDeep functions (cf CSBDeep getting started in https://github.com/CSBDeep/CSBDeep) to compute the CARE model.


About
-----
This tool uses modified code of picasso (https://github.com/jungmannlab/picasso) for spot localization and modified patches creation function from CSBDeep (https://github.com/CSBDeep/CSBDeep).

Tools developed by Nicolas Riss under the supervison of Julien Godet.
