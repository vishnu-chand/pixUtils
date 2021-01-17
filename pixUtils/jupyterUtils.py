def unwantedThings():
    # ################################################ clear gdrive trash #################################################
    # !df -h
    # !rm -rf ~/.local/share/Trash/*
    # !df -h

    # from google.colab import auth
    # from tqdm.notebook import tqdm
    # from pydrive.auth import GoogleAuth
    # from pydrive.drive import GoogleDrive
    # from oauth2client.client import GoogleCredentials

    # auth.authenticate_user()
    # gauth = GoogleAuth()
    # gauth.credentials = GoogleCredentials.get_application_default()
    # my_drive = GoogleDrive(gauth)

    # def deleteDriveTrash(book=''):
    #     query = "trashed=true"
    #     if book:
    #         query = f"title = '{book}' and trashed=true"
    #     for a_file in tqdm(my_drive.ListFile({'q': query}).GetList()):
    #         # print(f'the file {a_file["title"]}, is about to get deleted permanently.')
    #         try:
    #             a_file.Delete()
    #         except:
    #             pass

    # deleteDriveTrash()
    # ################################################ clear gdrive trash #################################################

    # ############################################# load 2nd drive #############################################
    # !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
    # !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
    # !apt-get update -qq 2>&1 > /dev/null
    # !apt-get -y install -qq google-drive-ocamlfuse fuse
    # from google.colab import auth
    # auth.authenticate_user()
    # from oauth2client.client import GoogleCredentials
    # creds = GoogleCredentials.get_application_default()
    # import getpass
    # !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
    # vcode = getpass.getpass()
    # !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
    # !mkdir -p /content/drive2
    # !google-drive-ocamlfuse /content/drive2
    # ############################################# load 2nd drive #############################################

    # !git clone https://github.com/tensorflow/models.git
    # dirop('/content/drive/My Drive/research/deeplab/trainBG', mv='/content/drive/My Drive/trainBG')
    # dirop('/content/drive/My Drive/research', rm=True)
    # dirop('/content/models/research/slim', mv='/content/drive/My Drive/research/slim')
    # dirop('/content/models/research/deeplab', mv='/content/drive/My Drive/research/deeplab')
    # dirop('/content/drive/My Drive/trainBG', mv='/content/drive/My Drive/research/deeplab/trainBG')
    # !rm -rf '/content/models'

    '''
    https://github.com/ZHKKKe/MODNet
    https://github.com/zhanghang1989/ResNeSt
    https://paperswithcode.com/paper/resnest-split-attention-networks

    !cd /content;git clone https://github.com/thuyngch/Human-Segmentation-PyTorch.git

    # https://www.kaggle.com/itsahmad/indoor-scenes-cvpr-2019
    # https://storage.googleapis.com/kaggle-data-sets/358221/702372/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20201128%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20201128T100831Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=0438f9d4ec37fa75a8467b1a62e48babcc32b7d9e3ba397b0ae29e74e0e2406b548c29ec811d5a1ee6e2b3fa495151301090b804bdd855e4ffcd48948c70a66af2e51eda79f9b7c60bb373f70f6e37dda787acad35a0910a868a818611c85a428ccc6eac1ce53f89b440fad54f64fa88e414a90f50eed78578ba6d358c58ed8d9f58a4e790d0b02b3393043b537b9a3c2bc804217c9eb42ff1442d48160125c0670c61aee1fc0f24bd66c0c713134c63c775aec1d789beac106b620d510bf019a18645abd8e3495ac4f6f05b963a48bcb3cff7b126dfb901710f9756d0dd349ad25940b78bece8c3bd614a5f7d46c4c825d51c5c7190935c5fb3383f18b01bf5
    https://zenodo.org/record/2654485/files/Indoor%20Object%20Detection%20Dataset.zip?download=1

    https://github.com/dong-x16/PortraitNet
    https://github.com/lizhengwei1992/Fast_Portrait_Segmentation


    https://colab.research.google.com/drive/10eGmnbXV-NVl-iSMwECwrSESob37V2kh#scrollTo=qjidyZ76WYeW
    '''


import tarfile
import numpy as np
import skimage.io as io
from copy import deepcopy
from pixUtils import *

try:
    from google.colab import files as coFile
    from google.colab.patches import cv2_imshow
except:
    pass


def releaseTf():
    try:
        import tensorflow as tf
        # device = cuda.get_current_device()
        # device.reset()
        tf.reset_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    except:
        pass


def reloadPy():
    def _is_module_deletable(modname, modpath):
        if modname.startswith('_cython_inline'):
            # Don't return cached inline compiled .PYX files
            return False
        for path in [sys.prefix]:
            if modpath.startswith(path):
                return False
        else:
            return set(modname.split('.'))

    """
    Del user modules to force Python to deeply reload them

    Do not del modules which are considered as system modules, i.e.
    modules installed in subdirectories of Python interpreter's binary
    Do not del C modules
    """
    log = []
    for modname, module in list(sys.modules.items()):
        modpath = getattr(module, '__file__', None)

        if modpath is None:
            # *module* is a C module that is statically linked into the
            # interpreter. There is no way to know its path, so we
            # choose to ignore it.
            continue

        if modname == 'reloader':
            # skip this module
            continue

        modules_to_delete = _is_module_deletable(modname, modpath)
        if modules_to_delete:
            log.append(modname)
            del sys.modules[modname]

    print(f"Reloaded modules: {log}")


def showImg(imgs, titles=None, figsize=(15, 15), bgrImg=True, cv=False):
    if type(imgs) != list:
        imgs = [imgs]
    if cv:
        for img in imgs:
            if not bgrImg:
                img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            cv2_imshow(img)
    else:
        plts = getSubPlots(len(imgs), figsize=figsize)
        if titles:
            for (title, ax), img, iname in zip(plts, imgs, titles):
                if bgrImg:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                title(iname)
        else:
            for (title, ax), img in zip(plts, imgs):
                if bgrImg:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                title(f"{img.dtype}{img.shape}")
        plt.show()


def quit():
    raise Exception('stoping execution')


def downloadDB(url, des, rename='', noCertificate=False, unzip=False):
    url = [c.strip() for c in url.replace('\n', ';').split(';')]
    url = ' '.join([c for c in url if c and not c.startswith('#')])
    dirop(des)
    downloadCmd = f'cd "{des}";'
    if not exists(url):
        if 'https://github.com' in url:
            downloadCmd += f'git clone "{url}";'
        elif 'https://drive.google.com' in url:
            if '/d/' in url:
                gid = url
                gid = gid.split('/d/')[1]
                gid = gid.split('/')[0]
            elif 'id=' in url:
                gid = url
                gid = gid.split('id=')[1]
                gid = gid.split('&')[0]
            downloadCmd += f'gdown https://drive.google.com/uc?id={gid};'
        elif 'https://www.youtube.com' in url:
            downloadCmd += f"youtube-dl '{url}' --print-json --restrict-filenames -o '%(id)s.%(ext)s'"
        elif noCertificate:
            downloadCmd += f'wget --no-check-certificate "{url}";'
        else:
            downloadCmd += f'wget "{url}";'
        old = set(glob(f'{des}/*'))
        print("____________________________________________________________________________________________________________________")
        print(f"\n             downloadCmd\n")
        print("____________________________________________________________________________________________________________________")
        os.system(downloadCmd)
        returnData = list(set(glob(f'{des}/*')) - old)
    else:
        returnData = [url]
    if len(returnData) != 1:
        print("skipping unzip no. file downloaded != 1")
        print("returnData:", returnData)
        return returnData
    returnData = returnData[0]
    if rename:
        returnData = dirop(returnData, mv=join(dirname(returnData), rename))
    if unzip:
        print("unziping dataset")
        print("zip: ", returnData)
        print("des: ", returnData.split('.')[0])
        returnData = unzipIt(returnData, returnData.split('.')[0])
    return returnData


def downloadVideo(vpath, compress):
    if type(vpath) == str:
        if compress:
            des = dirop(f"/tmp/{basename(vpath)}")
            os.system(f"ffmpeg -i {vpath} -vcodec libx265 -crf 28 {des} -y")
        else:
            des = vpath
    else:
        if compress:
            des = dirop(f"/tmp/{getTimeStamp()}")
            for v in vpath:
                print("compressing", v)
                summaryPath = f"{des}/{basename(v)}"
                os.system(f"ffmpeg -i {v} -vcodec libx265 -crf 28 {summaryPath} -y")
        else:
            des = vpath
        des = zipIt(des, f'/tmp/{basename(des)}.zip')
    print("des", des)
    coFile.download(des)
    return des