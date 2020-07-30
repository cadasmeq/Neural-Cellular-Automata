import imageio

def jpgs_to_gif(path):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(path, "movie.gif"), images)
    return "Done"


path = "../training_output"
jpgs_to_gif(path)