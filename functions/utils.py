import os
import imageio

png_dir = './training_output/steps_epoch_999'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))

print(images)
#imageio.mimsave('./movie_999.gif', images)