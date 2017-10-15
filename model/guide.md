# Guide & issues

1. Run 'run_linux.sh' in main 'MachineVision' folder to render image data.
2. Modify myscript.py to generate more or less images and to update the resolution.
  - The network architecture itself will update according to the number of classes and width/height of images.
3. Copy the render data directory system path and the labels_images.csv ( generated with myscript.py ) system path to
'ap_run_linux.sh' so that the correct image files index and labels are used.
4. Run ap_run_linux.sh to load the image data into numpy arrays and consequently train a convoluted neural network.

## TODO:

- Find out why memory runs out with larger/odd image sizes
- How to avoid memory issues
- How to avoid crashes with odd image sizes such as 50x50 ( while 100x100, 200x200 and 800x800 seemed fine )

## Settings notes:

- For now the settings of an image of 100x100 pixels work.
  - Not sure why 50x50 crashed, why 200x200 gave memory errors and why 800x800 worked?

## Network shapes explained:

The numerous pooling and convolution layers each take half of the original image size.

For the Mnist example it was: 28 / 2 = 14 / 2 = 7,
which is how it got to the number 3136: 7 * 7 * 64.

That means the convoluted network layers are calculated as follows:
image_width / 2 / 2.

## Errors:

- Memory leak, full memory stays occupied on gpus
  - check memory usage after restart with: ```watch -n 1 nvidia-smi```
- Make sure to let the program finish otherwise the memory is not freed and I have to restart the computer.
