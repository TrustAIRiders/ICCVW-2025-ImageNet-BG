import modules.backgroundSelector as bgS
import modules.backgroundSubstitutor as imG
import modules.miscBackgroundGenerator as bgG
import arguably
"""Background Substitution Toolkit"""

@arguably.command(alias="m")
def miscbg(path: str | None = None):
    """
    generates our selection of simple backgrounds
    :param path: destination directory
    """
    if not path[-1] == "/":
        path = path + "/"
    bgG.gen_and_save_suite(path)

@arguably.command(alias="s1")
def selbgm1():
    """
    the first method of background selection from a large dataset
    """
    bgS.method1()

@arguably.command(alias="s2")
def selbgm2(patho:str, pathi:str, *classes:str):
    """
    the second method of background selection from a large dataset (it may take some time)
    :param pathi: path containing the dataset to be scanned for backgrounds
    :param patho: path to save the images in
    :param classes: list of classes to scan (the names must be the same as the folder containing a class)
    """
    if not pathi[-1] == "/":
        pathi = pathi + "/"
    if not patho[-1] == "/":
        patho = patho + "/"
    bgS.method2(patho, pathi, classes)

@arguably.command(alias="s3")
def selbgm3(patho:str, pathi:str, number:int | None = None):
    """
    the second method of background selection from a large dataset (it may take some time)
    :param pathi: path containing the dataset to be scanned for backgrounds
    :param patho: path to save the images in
    :param number: number of instances to select (must be lower than the actual size of the used dataset)
    """
    if not pathi[-1] == "/":
        pathi = pathi + "/"
    if not patho[-1] == "/":
        patho = patho + "/"
    bgS.method3(patho, pathi, number)

@arguably.command(alias="g")
def generate( pathi:str, pathm:str, pathbg:str , patho:str | None = None):
    """
    generate the dataset with altered backgrounds based on ImageNet-S
    :param pathi: path containing the image data
    :param pathm: path containing the masking data (the internal structure of these two folders should be identical)
    :param pathbg: path containing the backgrounds to be used
    :param patho: path to save the images in
    """
    if not pathi[-1] == "/":
        pathi = pathi + "/"
    if not pathm[-1] == "/":
        pathm = pathm + "/"
    if not pathbg[-1] == "/":
        pathbg = pathbg + "/"
    if not patho[-1] == "/":
        patho = patho + "/"
    imG.generate_images(pathi, pathm, patho)

if __name__ == "__main__":
    arguably.run()