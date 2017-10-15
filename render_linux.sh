#!/bin/bash
# blender <blender file> <render in background> <pass a python script> <script file path> <script arguments: width, height>
blender basic_shape_gen.blend --background --python render_script.py 200 200 /output/path/render_gen
