# Blender Guide for Data Generation

Make sure to add Blender to bash profile to run it from command line, otherwise Python output is not visible.

In user home, .bash_profile:
`alias blender=/Applications/blender.app/Contents/MacOS/blender`

Assumes Blender is installed on mac, but will be similar setting for Linux.

## 1

Create a Python script <name.py> in the 'text editor'.

## 2

Import Blender Python to access Blender data, such as objects, scenes, frames and methods: ```import bpy```

## 3

Entire process is written in script, except for placing objects and adding randomized animation.

Blender uses Python 3.

Python scripts can be run with commandline with Blender.

## 4

Execute script with Blender:
```sh
#!/bin/bash
/Applications/blender.app/Contents/MacOS/blender basic_shape_gen.blend --background --python myscript.py
```
