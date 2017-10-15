# ImageGen

Program to generate images from a 3D scene with objects. Detects each object and randomly sets the display value. This value is stored in a '1-hot-vector' for use as feedback in a convolutional neural network. This project is meant to showcase how useful Blender can be to generate synthetic data for testing neural networks.

## Requirements:

- Blender 2.78
- Python 2 and 3
- OpenCV or PIL

## Paper

See ```SyntheticData_28_7_2017.pdf``` for an extensive explanation behind the concept of synthetic generation for training neural networks.

## Guide

First check ```basic_shape_gen.blend``` to inspect or modify the actual contents of the scene. Then modify and run ```render.sh``` to render an imagedata set and generate a ```.csv``` reference file. When rendering is complete, check out the 'model' directory, which contains the convolutional model and an input data processing script.

Check the filepaths so that you can easily input the render output into the training program.

An interesting test is to add different objects to the scene, and animate more parameters to determine wether this basic network is able to distinct the same objects.

## Examples

See example files ```labels_files.csv``` and ```labels_files_2.csv``` on what the render program outputs. Changing the objects in the scene, and adding 'ai_' to the object name will have it included by the render script in the labels and randomized display.

## License

See LICENCE
