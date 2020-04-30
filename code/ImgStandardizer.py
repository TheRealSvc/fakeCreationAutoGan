import PIL
from PIL import Image
import numpy as np
import os


#  get the image size of common_res with pixel number closest to in_size
def calc_common_size(in_size, common_res):
    pix_diff = { k: abs((np.prod(v)-np.prod(in_size))[(np.prod(v)-np.prod(in_size))<0]) for k, v in common_res.items()}
    min_pix_key = min(pix_diff, key=pix_diff.get)
    return common_res[min_pix_key] , min_pix_key


def resize_image(img, pixdims):
    img=img.resize(pixdims, PIL.Image.NEAREST)  # comment svc: adjust method from nearest to ... ?
    data = np.asarray(img, dtype="float32")
    return data


def create_npy_from_image(images_folder, outdir , output_prefix, common_res):
    allfiles = os.listdir(images_folder)
    jpegs1 = list(filter(lambda x: ".JPG" in x, allfiles))
    jpegs2 = list(filter(lambda x: ".jpg" in x, allfiles))
    jpegs = jpegs1 + jpegs2
    num_jpegs = len(jpegs)

# create all the folders if not existing
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        res_classes = ["/" + s for s in list(common_res.keys())]
        [os.mkdir(outdir+fol) for fol in res_classes ]

    [ os.mkdir(outdir+fol) for fol in list(common_res.keys()) if fol not in os.listdir(outdir)]
# initialize empty lists to hold the np images for standardizes sizes
    for k, v in common_res.items():
        exec(str(k)+"=[]")
        #my_dict[k] = []

# fill lists with standardized images
    for i, filename in enumerate(jpegs):
        print(i)
        img = Image.open(images_folder + filename)
        target_size, res_class = calc_common_size(img.size, common_res)
        data = resize_image(img, target_size)
        print(res_class)
        exec(str(res_class)+".append(data)")
        #my_dict[res_class] = my_dict[res_class].append(data) # vielleicht später mal, um execs mit nested dicts zu ersetzen

# stack images into arrays (one dimension added)
    for k, v in common_res.items():
        print(k)
        leni = eval("len("+k+")")
        if leni > 0:
            exec(str(k)+"_array=np.stack("+str(k)+")")
            f = eval("open('"+outdir + k + "/" + output_prefix + ".npy', 'wb')")
            exec("np.save('"+outdir + k + "/" + output_prefix + ".npy',"+ str(k)+"_array)")
            f.close()

# note: common_res and paths are currently not parameterized --> need to modify here
if __name__ == '__main__':
    common_res = {'real_std_128x128': (128, 128), 'real_std_256x256': (256, 256), 'real_std_640x480': (640, 480), 'real_std_800x600': (800, 600), 'real_std_1024x768': (1024, 768),'real_std_1600x1200': (1600, 1200)}
    create_npy_from_image(images_folder="E:/myAWS/photographSelection_real/selection/", outdir="E:/myAWS/photographSelection_real/standard/", output_prefix="samp1", common_res=common_res)

