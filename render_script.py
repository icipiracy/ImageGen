import bpy
import random
import sys
import csv
import os
import platform
import numpy

"""

TODO:
- update blender file to include more different objects and variable background.

"""

# # Image width and height:
# res = sys.argv[5]
# samples = sys.argv[6]
#
# print(res,samples)

res = int(sys.argv[5])
samples = int(sys.argv[6])

def zeroes(n):
    listofzeros = [0] * n
    return listofzeros

def get_os():

    """

    Test to get operating system information:

    """

    print(os.name)
    print(platform.system())
    print(platform.release())

    return


def run_data_gen():

    os_type = platform.system()

    # Path to output images:
    fp = sys.argv[7]

    print(os.path.dirname(os.path.abspath(__file__)))

    bpy.context.scene.frame_set(0)

    def list_ai_detectable_objects():

        air = []

        for obj in bpy.data.objects:
            if 'ai' in obj.name:
                print(obj.name)
                air.append(obj)

        return air

    ai_objects = list_ai_detectable_objects()
    labels_files = []

    def write_dataset(labels_files):

        # write labels_files to csv

        filepath = "labels_files_2.csv"

        # lbcsv = csv(labels_files) encode labels_files to csv

        with open(filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerows(labels_files)

        return

    def gen_lbf(label,fname,inames):

        # label must be array
        # fname must be unique filename string

        labels_files.append([label,fname,inames])
        return labels_files

    def update_scene(x):

        label = zeroes(len(ai_objects))

        print("Length ai_objects zeroes:")
        print(len(ai_objects))
        print(len(label))

        inames = []
        fname = str(x) + "_gen.png"

        bpy.context.scene.render.filepath = fp + fname

        sx = random.randint(0,len(label) - 1)

        print("Random selector:")
        print(sx)

        label[sx] = 1

        for idx,val in enumerate(ai_objects):

            if idx == sx:
                cstate = True
            else:
                exsx = random.randint(0,1)
                if exsx == 0:
                    cstate = False
                else:
                    cstate = True

            val.cycles_visibility.transmission = cstate
            val.cycles_visibility.camera = cstate
            val.cycles_visibility.diffuse = cstate
            val.cycles_visibility.glossy = cstate
            val.cycles_visibility.scatter = cstate
            val.cycles_visibility.shadow = cstate

            inames.append(val.name)

        return label,fname,inames

    def gen_render_data(size):

        # Size is int of amount of images to generate.
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.render.resolution_x = res
        bpy.context.scene.render.resolution_y = res

        for x in range(size):
            labels_files = gen_lbf(*update_scene(x))
            # Render command, use still render options:
            bpy.ops.render.render( write_still=True )
            bpy.context.scene.frame_set(x)

            # Write to file every step, so that rendering can be stopped at any time:
            write_dataset(labels_files)

        return labels_files

    data_set_size = 100

    # Change int to actual required data set later:
    gen_render_data(data_set_size)

    return

run_data_gen()
