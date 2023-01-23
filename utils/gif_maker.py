import os
import re
import imageio


def make_gif(folder_path, gif_name):
    # Get all jpeg files in the folder
    jpeg_files = [f for f in os.listdir(folder_path) if f.endswith('.jpeg')]

    # Split the files into two groups: those with numbers in the name and those without
    files_with_numbers = []
    files_without_numbers = []
    for f in jpeg_files:
        if re.search(r'\d', f):
            files_with_numbers.append(f)
        else:
            files_without_numbers.append(f)

    # Sort the files with numbers by the number in the file name
    files_with_numbers.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    # Create the gif using imageio
    images = [imageio.imread(os.path.join(folder_path, f))
              for f in files_with_numbers + files_without_numbers]
    durations = [0.2 for f in files_with_numbers] + \
        [1.0 for f in files_without_numbers]
    imageio.mimsave(gif_name, images, duration=durations)

folder_path = './'
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
for subfolder in subfolders:
    gif_name = os.path.basename(subfolder) + '.gif'
    make_gif(subfolder, gif_name)
