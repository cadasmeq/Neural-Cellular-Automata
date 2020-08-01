import os
import imageio
import pandas as pd

# There is a mini-help guide bottom if you have troubles understanding the code :)

def pngs2gif(images_folder, export_to=None, file_name=None):

    
    default_export = "./gifs"

    # Check if export path exits
    if export_to == None:
        export_to = default_export
        
    if not os.path.exists(export_to):
        os.mkdir(export_to)
    
    # Create gifs name and generate export path
    images = []
    if file_name == None:
        gif_name = str(os.path.split(images_folder)[-1]) + ".gif"   
    else:
        gif_name = file_name
    output_path = os.path.join(export_to, gif_name)

    # Appending sorted png's files to images list
    for file in sorted(os.listdir(images_folder), key=len):
        if file.endswith('.png'):
            file_path = os.path.join(images_folder, file)
            images.append(imageio.imread(file_path))

    # exporting gif
    imageio.mimsave(output_path, images)
    print("Gif created in {}".format(output_path))

png_folder = './training_output/steps_epoch_999'
gif_folder = "./training_output/gifs"
pngs2gif(png_folder, export_to=gif_folder, file_name="fail2_owl_epoch_999.gif")
 
def create_df(weights_path):
        
    model_name = 'NCA'
    uid = 1
    epoch = 100
    loss = 0.325
    lr = 2e-3
    r_seed = 24
    r_steeps = 24

    data = {'id':[uid],
            'model_name':[model_name],
            'epoch':[epoch],
            'loss':[loss],
            'learning_rate':[lr],
            'r_seed':[r_seed],
            'r_steps':[r_steps],
            }

    df = pd.DataFrame(data, columns=['id', 'model_name', 'epoch', 'loss', 'learning_rate'])
    print(df)


    '''
    -------------------------------------------------------------------
                            HOW TO: pngs2gif()
    -------------------------------------------------------------------
    Params: 
    1) images_folder:   Input path that contains all images .png
    2) export_to:       Output path where gif file will be exported.
    3) file_name:       Name gifs file will have.
    -------------------------------------------------------------------
    Notes:
    1) images_folder: You can specify this param utter raw, function will sort it by name asc.
    2) export_to: If this param didn't be set, by default, output path will be "./gifs".
    3) file_name: If this param didn't be set, by default, the gif's name will be the same of the folder that contains the input images
    -------------------------------------------------------------------
    Example 1):
    input = './outputs/sequences'
    export_folder = './myPath'
    gif_name = 'seq.gif'
    
    pngs2gif(input, 
             export_to=export_folder, 
             file_name=gif_name)

    output >> "./myPath/seq.gif"
     -------------------------------------------------------------------
    Example 2)
    input = './outputs/sequences'
    
    pngs2gif(input)
    output >> "./gifs/sequences.gif"
    -------------------------------------------------------------------
    '''