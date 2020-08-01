import os
import imageio
import pandas as pd

png_dir = './training_output/steps_epoch_333'

def pngs_to_gif(png_dir, name=None):
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            images.append(file_name)
    
    print(sorted(images))



    #         file_path = os.path.join(png_dir, file_name)
    #         images.append(imageio.imread(file_path))

    # out_path = os.path.join(png_dir, "movie.gif")
    # imageio.mimsave(out_path, images)

    # print("Gif created in {}".format(out_path))

pngs_to_gif(png_dir)

def create_df(weights_path):
        
    model_name = 'NCA'
    uid = 1
    epoch = 100
    loss = 0.325
    lr = 2e-3

    data = {

        'id':[uid],
        'model_name':[model_name],
        'epoch':[epoch],
        'loss':[loss],
        'learning_rate':[lr]
    }

    df = pd.DataFrame(data, columns=['id', 'model_name', 'epoch', 'loss', 'learning_rate'])
    print(df)