import re
import ast
import time
import json
import pickle
import argparse
from itertools import groupby
from itertools import zip_longest
from itertools import permutations
from itertools import combinations
from .pixCommon import *
from .bashIt import *

try:
    import yaml
except:
    pass

try:
    from matplotlib import pyplot as plt
except:
    pass

try:
    import dlib
except:
    pass

try:
    import pandas as pd

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
except:
    pass


class tqdm:
    """
    fix: notbook extra line issue
    """
    from tqdm import tqdm

    def __init__(self, iterable=None, desc=None, total=None, leave=True, file=sys.stdout, ncols=None, mininterval=0.1, maxinterval=10.0,
            miniters=None, ascii=None, disable=False, unit='it', unit_scale=False, dynamic_ncols=False, smoothing=0.3,
            bar_format=None, initial=0, position=None, postfix=None, unit_divisor=1000, write_bytes=None, lock_args=None, gui=False, **kwargs):
        self.args = dict(
                iterable=iterable, desc=desc, total=total, leave=leave, file=file, ncols=ncols, mininterval=mininterval, maxinterval=maxinterval,
                miniters=miniters, ascii=ascii, disable=disable, unit=unit, unit_scale=unit_scale, dynamic_ncols=dynamic_ncols, smoothing=smoothing,
                bar_format=bar_format, initial=initial, position=position, postfix=postfix, unit_divisor=unit_divisor, write_bytes=write_bytes, lock_args=lock_args, gui=gui)
        self.args.update(kwargs)
        self.pbar = None

    def __iter__(self):
        iterable = self.args['iterable']
        with self.tqdm(**self.args) as self.pbar:
            for i in iterable:
                yield i
                self.pbar.update(1)

    def set_description(self, msg):
        self.pbar.set_description(msg)


class DotDict(dict):
    def __init__(self, datas=None):
        super().__init__()
        if isinstance(datas, argparse.Namespace):
            datas = vars(datas)
        datas = dict() if datas is None else datas
        for k, v in datas.items():
            self[k] = v

    def __getattr__(self, key):
        if key not in self:
            print("56 __getattr__ pixCommon key: ", key)
            raise AttributeError(key)
        else:
            return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        keys = list(self.keys())
        nSpace = len(max(keys, key=lambda x: len(x))) + 2
        keys = sorted(keys)
        data = [f'{key:{nSpace}}: {self[key]},' for key in keys]
        data = '{\n%s\n}' % '\n'.join(data)
        return data

    def copy(self):
        return DotDict(super().copy())

    def toJson(self):
        res = OrderedDict()
        for k, v in self.items():
            try:
                json.dumps({k: v})
                res[k] = v
            except:
                res[k] = str(v)
        return json.dumps(res)

    def toDict(self):
        res = OrderedDict()
        for k, v in self.items():
            try:
                json.dumps({k: v})
                res[k] = v
            except:
                res[k] = str(v)
        return res


def readYaml(src, defaultDict=None):
    data = defaultDict
    if os.path.exists(src):
        with open(src, 'r') as book:
            data = yaml.safe_load(book)
    return DotDict(data)


# def writeYaml(yamlPath, jObjs):
#     with open(yamlPath, 'w') as book:
#         yaml.dump(yaml.safe_load(jObjs), book, default_flow_style=False, sort_keys=False)


def readPkl(pklPath, defaultData=None):
    if not os.path.exists(pklPath):
        print("loading pklPath: ", pklPath)
        return defaultData
    return pickle.load(open(pklPath, 'rb'))


def writePkl(pklPath, objs):
    pickle.dump(objs, open(dirop(pklPath), 'wb'))


def dir2(var):
    """
    list all the methods and attributes present in object
    """
    for v in dir(var):
        print(v)
    print("34 dir2 common : ", )
    quit()


# def checkAttr(obj, b, getAttr=False):
#     a = set(vars(obj).keys())
#     if getAttr:
#         print(a)
#     extra = a - a.intersection(b)
#     if len(extra):
#         raise Exception(extra)


def bboxLabel(img, txt="", loc=(30, 45), color=(255, 255, 255), thickness=3, txtSize=1, txtFont=cv2.QT_FONT_NORMAL, txtThickness=3, txtColor=None, asTitle=None):
    if len(loc) == 4:
        x0, y0, w, h = loc
        x0, y0, rw, rh = int(x0), int(y0), int(w), int(h)
        cv2.rectangle(img, (x0, y0), (x0 + rw, y0 + rh), list(color), thickness)
    else:
        if asTitle is None:
            asTitle = True
        x0, y0, rw, rh = int(loc[0]), int(loc[1]), 0, 0
    txt = str(txt)
    if txt != "":
        if txtColor is None:
            txtColor = (0, 0, 0)
        (w, h), baseLine = cv2.getTextSize(txt, txtFont, txtSize, txtThickness)
        # baseLine -> to fit char like p,y in box
        if asTitle:
            h, w = img.shape[:2]
            zimg = np.zeros([60, w, 3], 'u1') + 255
            cv2.putText(zimg, txt, (x0, y0 + rh - baseLine), txtFont, txtSize, txtColor, txtThickness, cv2.LINE_AA)
            img = cv2.vconcat([zimg, img])
        else:
            cv2.rectangle(img, (x0, y0 + rh), (x0 + w, y0 + rh - h - baseLine), color, -1)
            cv2.putText(img, txt, (x0, y0 + rh - baseLine), txtFont, txtSize, txtColor, txtThickness, cv2.LINE_AA)
    return img


def drawText(img, txt, loc, color=(255, 255, 255), txtSize=1, txtFont=cv2.FONT_HERSHEY_SIMPLEX, txtThickness=3, txtColor=None):
    (w, h), baseLine = cv2.getTextSize(txt, txtFont, txtSize, txtThickness)
    x0, y0 = int(loc[0]), int(loc[1])
    if txtColor is None:
        txtColor = (0, 0, 0)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 - h - baseLine), color, -1)
    cv2.putText(img, txt, (x0, y0 - baseLine), txtFont, txtSize, txtColor, txtThickness)
    return img


def frameFit(img, bbox):
    """
    ensure the bbox will not go away from the image boundary
    """
    imHeight, imWidht = img.shape[:2]
    x0, y0, width, height = bbox
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = x0 + int(width), y0 + int(height)
    x1, y1 = min(x1, imWidht), min(y1, imHeight)
    return np.array((x0, y0, max(0, x1 - x0), max(0, y1 - y0)))


def bboxScale(img, bbox, scaleWH):
    try:
        sw, sh = scaleWH
    except:
        sw, sh = scaleWH, scaleWH
    x, y, w, h = bbox
    xc, yc = (x + w / 2, y + h / 2)
    w *= sw
    h *= sh
    x, y = xc - w / 2, yc - h / 2
    return frameFit(img, (x, y, w, h))


def putSubImg(mainImg, subImg, loc, interpolation=cv2.INTER_CUBIC):
    """
    place the sub image inside the genFrame image
    """
    if len(loc) == 2:
        x, y = int(loc[0]), int(loc[1])
        h, w = subImg.shape[:2]
    else:
        x, y, w, h = int(loc[0]), int(loc[1]), int(loc[2]), int(loc[3])
        subImg = cv2.resize(subImg, (w, h), interpolation=interpolation)
    x, y, w, h = frameFit(mainImg, (x, y, w, h))
    mainImg[y:y + h, x:x + w] = getSubImg(subImg, (0, 0, w, h))
    return mainImg


def getSubImg(im1, bbox):
    """
    crop sub image from the given input image and bbox
    """
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    img = im1[y:y + h, x:x + w]
    if img.shape[0] and img.shape[1]:
        return img


def maskIt(roi, roiMask):
    """
    apply mask on the image. It can accept both gray and colors image
    """
    if len(roi.shape) == 3 and len(roiMask.shape) == 2:
        roiMask = cv2.cvtColor(roiMask, cv2.COLOR_GRAY2BGR)
    elif len(roi.shape) == 2 and len(roiMask.shape) == 3:
        roiMask = cv2.cvtColor(roiMask, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(roi, roiMask)


# def imHconcat(imgs, sizeRC, interpolation=cv2.INTER_LINEAR):
#     rh, rw = sizeRC[:2]
#     res = []
#     for queryImg in imgs:
#         qh, qw = queryImg.shape[:2]
#         queryImg = cv2.resize(queryImg, (int(rw * qw / qh), int(rh)), interpolation=interpolation)
#         res.append(queryImg)
#     return cv2.hconcat(res)
#
#
# def imVconcat(imgs, sizeRC, interpolation=cv2.INTER_LINEAR):
#     rh, rw = sizeRC[:2]
#     res = []
#     for queryImg in imgs:
#         qh, qw = queryImg.shape[:2]
#         queryImg = cv2.resize(queryImg, (int(rw), int(rh * qh / qw)), interpolation=interpolation)
#         res.append(queryImg)
#     return cv2.vconcat(res)


class VideoWrtier:
    """mjpg xvid mp4v"""

    def __init__(self, path, camFps, size=None, codec='mp4v'):
        self.path = path
        try:
            self.fps = camFps.get(cv2.CAP_PROP_FPS)
        except:
            self.fps = camFps
        self.__vWriter = None
        self.__size = size
        self.__codec = cv2.VideoWriter_fourcc(*(codec.upper()))
        print("writing :", path, '@', self.fps, 'fps')

    def write(self, img):
        if self.__vWriter is None:
            if self.__size is None:
                self.__size = tuple(img.shape[:2])
            self.__vWriter = cv2.VideoWriter(self.path, self.__codec, self.fps, self.__size[::-1])
        if tuple(img.shape[:2]) != self.__size:
            img = cv2.resize(img, self.__size[::-1])
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        self.__vWriter.write(img)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.__vWriter:
            self.__vWriter.release()
        else:
            print(f"video: {self.path} closed without opening")

    def __enter__(self):
        return self


class clk:
    def __init__(self, roundBy=4):
        self.__roundBy = roundBy
        self.__toks = [["start", dt.now()]]

    def tok(self, name):
        self.__toks.append([name, dt.now()])
        return self

    def __repr__(self):
        cols, datas = self.get(useNumpy=False)
        try:
            datas = pd.DataFrame(datas, columns=cols).round(self.__roundBy)
        except:
            print(cols)
            datas = np.array(datas)
            datas[:, 1:] = datas[:, 1:].astype(float).round(self.__roundBy)
            datas = datas.astype(str)
        return str(datas)

    def get(self, useNumpy=True):
        toks = self.__toks
        _, stik = toks[0]
        datas = []
        cols = 'name', 'splitFps', 'splitSec', 'totalSec', 'totalFps'
        for (_, tik), (name, tok) in zip(toks, toks[1:]):
            lap = self.__getLap(tik, tok)
            tlap = self.__getLap(stik, tok)
            data = name, 1 / lap, lap, tlap, 1 / tlap
            datas.append(data)
        if useNumpy:
            datas = np.array(datas)
        return cols, datas

    def last(self, roundBy=None):
        roundBy = roundBy or self.__roundBy
        toks = self.__toks
        (_, tik), (_, tok) = toks[-2], toks[-1]
        lap = self.__getLap(tik, tok)
        return round(lap, roundBy)

    @staticmethod
    def __getLap(tik, tok):
        lap = tok - tik
        lap = lap.seconds + (lap.microseconds / 1000000)
        return lap


class Wait:
    def __init__(self):
        self.pause = False

    def __call__(self, delay=1):
        if self.pause:
            delay = 0
        key = cv2.waitKey(delay)
        if key == 32:
            self.pause = True
        if key == 13:
            self.pause = False
        return key


__wait = Wait()


def showImg(winname='output', imC=None, delay=None, windowConfig=0, nRow=None, chFirst=False):
    winname = str(winname)
    if imC is not None:
        if type(imC) is not list:
            imC = [imC]
        imC = photoframe(imC, nRow=nRow, chFirst=chFirst)
        cv2.namedWindow(winname, windowConfig)
        cv2.imshow(winname, imC)

    if delay is not None:
        key = __wait(delay)
        return key
    return imC


# def pshowImg(winname=None, imC=None, delay=0):
#     winname = str(winname)
#     if imC is not None:
#         if type(imC) is list:
#             pass
#         plt.imshow(imC)
#     if delay is not None:
#         if delay == 0:
#             plt.show()
#         # else:
#         #     plt.pause(delay / 1000)
#         return 1
#     return imC

def prr(name, img):
    import torch  # TODO remove this or add tf support
    if type(img) == list:
        img = torch.stack(img)
    try:
        mean, std = f'\n\t\tmean :\t{img.mean()}', f'\n\t\tstd  :\t{img.std()}'
    except:
        mean, std = '', ''
    data = f"""
    ________________ {name} ________________
            shape:\t{img.shape}
            dtype:\t{img.dtype}
            min  :\t{img.min()}
            max  :\t{img.max()}{mean}{std}
            """
    print(data.strip())


def getSubPlots(nrows=1, ncols=-1, axisOff=True, tight=True, figsize=(15, 15), sharex=False, sharey=False, squeeze=True, subplot_kw=None, gridspec_kw=None, **fig_kw):
    if ncols != -1:
        nimgs = nrows * ncols
    else:
        nimgs = nrows
        nrows = int(np.ceil(nimgs ** .5))
        ncols = int(nimgs / nrows)
        ncols = ncols + 1 if (nrows * ncols) < nimgs else ncols

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex, sharey=sharey, squeeze=squeeze, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)

    if nrows == 1 and ncols == 1:
        axs = np.array([axs])
    axs = axs.ravel()
    for ix, ax in enumerate(axs):
        if ix < nimgs:
            yield lambda x: ax.set_title(x, color='white'), ax
        if axisOff:
            ax.axis('off')
    if tight:
        plt.tight_layout()
    if axisOff:
        plt.axis('off')


def zipIt(src, desZip, rm=False):
    if not exists(src):
        raise Exception(f'''Fail src: {src} \n\tnot found''')
    if exists(desZip):
        if rm:
            os.remove(desZip)
        else:
            raise Exception(f'''Fail des: {desZip} \n\talready exists delete it before operation''')
    desZip, zipExt = os.path.splitext(desZip)
    if os.path.isfile(src):
        tempDir = join(dirname(src), getTimeStamp())
        if os.path.exists(tempDir):
            raise Exception(f'''Fail tempDir: {tempDir} \n\talready exists delete it before operation''')
        os.makedirs(tempDir)
        shutil.copy(src, tempDir)
        desZip = shutil.make_archive(desZip, zipExt[1:], tempDir)
        shutil.rmtree(tempDir, ignore_errors=True)
    else:
        desZip = shutil.make_archive(desZip, zipExt[1:], src)
    return desZip


def unzipIt(src, desDir, rm=False):
    if not exists(src):
        raise Exception(f'''Fail src: {src} \n\tnot found''')
    if os.path.splitext(desDir)[-1]:
        raise Exception(f'''Fail desDir: {desDir} \n\tshould be folder''')
    tempDir = join(dirname(desDir), getTimeStamp())
    shutil.unpack_archive(src, tempDir)
    if not exists(desDir):
        os.makedirs(desDir)
    for mvSrc in os.listdir(tempDir):
        mvSrc = join(tempDir, mvSrc)
        mvDes = join(desDir, basename(mvSrc))
        if rm is True and exists(mvDes):
            if os.path.isfile(mvDes):
                os.remove(mvDes)
            else:
                shutil.rmtree(mvDes, ignore_errors=True)
        try:
            shutil.move(mvSrc, desDir)
        except Exception as exp:
            shutil.rmtree(tempDir, ignore_errors=True)
            raise Exception(exp)
    shutil.rmtree(tempDir, ignore_errors=True)
    return desDir


# def float2img(img, pixmin=0, pixmax=255, dtype=0):
#     '''
#     convert oldFeature to (0 to 255) range
#     '''
#     return cv2.normalize(img, None, pixmin, pixmax, 32, dtype)


def float2img(img, min=None, max=None):
    min = img.min() if min is None else min
    max = img.max() if max is None else max
    img = img.astype('f4')
    img -= min
    img /= max
    return (255 * img).astype('u1')


def photoframe(imgs, rcsize=None, nRow=None, resize_method=cv2.INTER_LINEAR, fit=False, asgray=False, chFirst=False):
    """
    # This method pack the array of images in a visually pleasing manner.
    # If the nCol is not specified then the nRow and nCol are equally divided
    # This method can automatically pack images of different size. Default stitch size is 128,128
    # when fit is True final photo frame size will be rcsize
    #          is False individual image size will be rcsize
    # Examples
    # --------
        video = Player(GetFeed(join(dbpath, 'videos', r'remove_rain.mp4')), custom_fn=None)
        for fnos, imgs in video.chunk(4):
            i1 = photoframe(imgs, nCol=None)
            i2 = photoframe(imgs, nCol=4)
            i3 = photoframe(imgs, nCol=4, rcsize=(200,300),nimgs=7)
            i4 = photoframe(imgs, nCol=3, nimgs=7)
            i5 = photoframe(imgs, nCol=4, rcsize=imgs[0].shape)
            i6 = photoframe(imgs, nCol=6, rcsize=imgs[0].shape, fit=True)
            i7 = photoframe(imgs, nCol=4, rcsize=imgs[0].shape, fit=True, asgray=True)
            for i, oldFeature in enumerate([i1, i2, i3, i4, i5, i6, i7], 1):
                print(i, oldFeature.shape)
                win('i%s' % i, )(oldFeature)
            win('totoal')(photoframe([i1, i2, i3, i4, i5, i6, i7]))
            if win().__wait(waittime) == 'esc':
                break
    """
    if len(imgs):
        if chFirst:
            imgs = np.array([np.transpose(img, [1, 2, 0]) for img in imgs])
        if rcsize is None:
            rcsize = imgs[0].shape
        imrow, imcol = rcsize[:2]  # fetch first two vals
        nimgs = len(imgs)
        nRow = int(np.ceil(nimgs ** .5)) if nRow is None else int(nRow)
        nCol = nimgs / nRow
        nCol = int(np.ceil(nCol + 1)) if (nRow * nCol) - nimgs else int(np.ceil(nCol))
        if fit:
            imrow /= nRow
            imcol /= nCol
        imrow, imcol = int(imrow), int(imcol)
        resshape = (imrow, imcol) if asgray else (imrow, imcol, 3)
        imgs = zip_longest(list(range(nRow * nCol)), imgs, fillvalue=np.zeros(resshape, imgs[0].dtype))
        resimg = []
        for i, imggroup in groupby(imgs, lambda k: k[0] // nCol):
            rowimg = []
            for i, img in imggroup:
                if img.dtype != np.uint8:
                    img = float2img(img)
                if asgray:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[-1] == 1:
                    img = img.reshape(*img.shape[:2])
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if tuple(img.shape) != resshape:
                    img = cv2.resize(img, (imcol, imrow), interpolation=resize_method)
                rowimg.append(img)
            resimg.append(cv2.hconcat(rowimg))
        return cv2.vconcat(resimg)


def downloadDB(url, des, rename=''):
    url = [c.strip() for c in url.replace('\n', ';').split(';')]
    url = ' '.join([c for c in url if c and not c.startswith('#')])
    dirop(des)
    downloadCmd = f'cd "{des}";'
    if not exists(url):
        if url.startswith('git+'):
            downloadCmd += f'git clone "{url.lstrip("git+")}";'
        elif url.startswith('gdrive+'):
            url = url.lstrip('gdrive+')
            if '/d/' in url:
                gid = url
                gid = gid.split('/d/')[1]
                gid = gid.split('/')[0]
            elif 'id=' in url:
                gid = url
                gid = gid.split('id=')[1]
                gid = gid.split('&')[0]
            downloadCmd += f'gdown https://drive.google.com/uc?id={gid};'
        elif url.startswith('youtube+'):
            downloadCmd += f"youtube-dl '{url.lstrip('youtube+')}' --print-json --restrict-filenames -o '%(id)s.%(ext)s'"
        elif url.startswith('wgetNoCertificate+'):
            downloadCmd += f'wget --no-check-certificate "{url.lstrip("wgetNoCertificate+")}";'
        elif url.startswith('wget+'):
            downloadCmd += f'wget "{url.lstrip("wget+")}";'
        old = set(glob(f'{des}/*'))
        print("____________________________________________________________________________________________________________________")
        print(f"\n             {downloadCmd}\n")
        print("____________________________________________________________________________________________________________________")
        exeIt(downloadCmd, returnOutput=False)
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
    try:
        print("unzip dataset")
        print("zip: ", returnData)
        returnData = unzipIt(returnData, f"{dirname(returnData)}/{basename(returnData).split('.')[0]}")
    except Exception as exp:
        print(f"skipping unzip: {returnData}")
        print(exp)
    print("returnData: ", returnData)
    return returnData


def replaces(path, *words):
    path = str(path)
    assert len(words) % 2 == 0
    words = zip(words[::2], words[1::2])
    for word in words:
        path = path.replace(*word)
    return path


def compareVersions(versions, compareBy, ovideoPlayer=None, putTitle=bboxLabel, bbox=None, showDiff=False):
    vPlayer = videoPlayer if ovideoPlayer is None else ovideoPlayer
    vpaths = [compareBy] + [version for version in versions if version != compareBy]
    vplayers = [vPlayer(cv2.VideoCapture(version)) for version in vpaths]
    for ix, data in enumerate(zip(*vplayers)):
        imgs = []
        for vpath, (fno, ftm, img) in zip(vpaths, data):
            img = imResize(img, (780, 1280))
            if bbox is True:
                winname = "select roi"
                cv2.namedWindow(winname, 0)
                bbox = cv2.selectROI(winname, img)
                cv2.destroyWindow(winname)
            if bbox is not None:
                img = getSubImg(img, bbox)
            # img = bboxLabel(img, basename(vpath))
            imgs.append(img)
        datas = []
        for ix, img in enumerate(imgs[1:], 1):
            res = []
            res.append(putTitle(imgs[0].copy(), basename(vpaths[0])))
            res.append(putTitle(imgs[ix].copy(), basename(vpaths[ix])))
            if showDiff:
                diff = cv2.absdiff(imgs[0], img)
                res.append(diff)
                res.append(cv2.inRange(diff.min(axis=-1), 10, 300))
            datas.append(res)
        yield datas


def video2img(vpath, des):
    for fno, ftm, img in videoPlayer(vpath):
        cv2.imwrite(des.format(fno=fno), img)


def rglob(p1, p2=None):
    if p2:
        res = []
        for p1 in glob(p1):
            res.extend(list(map(str, Path(p1).rglob(p2))))
        return res
    else:
        return glob(p1)
