# ImageGen

Program to generate images from a 3D scene with objects. Detects each object and randomly sets the display value. This value is stored in a '1-hot-vector' for use as feedback in a convolutional neural network. This project is meant to showcase how useful Blender can be to generate synthetic data for testing neural networks.

## Requirements:

- Blender 2.78
- Python 2 and 3

## Guide

First check ```basic_shape_gen.blend``` to inspect or modify the actual contents of the scene. Then modify and run ```render_linux.sh``` to render an imagedata set and generate a ```.csv``` reference file. When rendering is complete, check out the 'model' directory, which contains the convolutional model and an input data processing script.

Check the filepaths so that you can easily input the render output into the training program.

An interesting test is to add different objects to the scene, and animate more parameters to determine wether this basic network is able to distinct the same objects.

## License

See LICENCE
