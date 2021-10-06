import os
import cv2
import sys
import shutil
import random
import logging
import argparse
import tempfile
import traceback
import numpy as np
from glob import glob
from os.path import join
from pathlib import Path
from os.path import exists
from os.path import dirname
from os.path import basename
from datetime import datetime as dt
from collections import OrderedDict
from collections import defaultdict

try:
    from PIL import Image
except:
    pass

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, formatter=dict(float=lambda x: "{0:8.4f}".format(x)))


def setSeed(seed=4487):
    # print(f"setting random seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except:
        print("skip setting torch seed")
    return seed


def getArgs(**kwArgs):
    def str2bool(val):
        if val.lower() in ('yes', 'true', 't', 'y', '1'):
            val = True
        elif val.lower() in ('no', 'false', 'f', 'n', '0'):
            val = False
        else:
            raise Exception(f"""
            unknown datatype: {val}
            expected type: ('yes', 'true', 't', 'y', '1'), ('no', 'false', 'f', 'n', '0')
            """)
        return val

    parser = argparse.ArgumentParser()
    for name, value in kwArgs.items():
        argType = type(value)
        if isinstance(value, bool):
            value = 'yes' if value else 'no'
            argType = str2bool
        parser.add_argument(f"--{name}", default=value, type=argType, help=f" eg: {name}={value}")
    return vars(parser.parse_known_args()[0])


def imResize(img, sizeRC=None, scaleRC=None, interpolation=cv2.INTER_LINEAR):
    if sizeRC is not None:
        r, c = sizeRC[:2]
    else:
        try:
            dr, dc = scaleRC
        except:
            dr, dc = scaleRC, scaleRC
        r, c = img.shape[:2]
        r, c = r * dr, c * dc
    if interpolation == 'aa':
        img = np.array(Image.fromarray(img).resize((int(c), int(r)), Image.ANTIALIAS))
    else:
        img = cv2.resize(img, (int(c), int(r)), interpolation=interpolation)
    return img


def getPath(p):
    p = f"{p}"
    for fn in [os.path.expandvars, os.path.expanduser, os.path.abspath]:
        p = fn(p)
    return p


def moveCopy(src, des, op, isFile, rm):
    des = getPath(des)
    desDir = dirname(des)
    if not rm and exists(des):
        raise Exception(f'''Fail des: {des}
                                    already exists delete it or try different name
                            eg: change dirop('{src}', cpDir='{desDir}', rm=False)
                                to     dirop('{src}', cpDir='{desDir}', rm=True)
                                or     dirop('{src}', cpDir='{desDir}', rm=False, desName='newName')
                        ''')
    if isFile:
        if rm and exists(des):
            os.remove(des)
        mkpath = dirname(des)
        if not exists(mkpath):
            os.makedirs(mkpath)
    else:
        if rm and exists(des):
            shutil.rmtree(des, ignore_errors=True)
    return op(src, des)


def dirop(path, *, mkdir=True, rm=False, isFile=None, cpDir=None, mvDir=None, desName=None):
    path = getPath(path)
    if isFile is None:
        isFile = os.path.splitext(path)[-1]
    if cpDir or mvDir:
        if not exists(path):
            raise Exception(f'''Fail src: {path}
                                            not found''')
    elif rm and exists(path):
        if isFile:
            os.remove(path)
        else:
            shutil.rmtree(path, ignore_errors=True)
    mkpath = dirname(path) if isFile else path
    if mkdir and not exists(mkpath) and mkpath:
        os.makedirs(mkpath)
    if cpDir:
        copy = shutil.copy if isFile else shutil.copytree
        desName = desName or basename(path)
        path = moveCopy(path, f"{cpDir}/{desName}", copy, isFile, rm=rm)
    elif mvDir:
        desName = desName or basename(path)
        path = moveCopy(path, f"{mvDir}/{desName}", shutil.move, isFile, rm=rm)
    return path


def getTimeStamp():
    return dt.now().strftime("%b%d_%H_%M_%S%f")


def videoPlayer(vpath, startSec=0.0, stopSec=np.inf):
    cam = vpath if type(vpath) == cv2.VideoCapture else cv2.VideoCapture(vpath)
    ok, ftm, fno = True, startSec, 0
    if ftm:
        cam.set(cv2.CAP_PROP_POS_MSEC, ftm * 1000)
    while ok:
        ok, img = cam.read()
        ok = ok and img is not None and ftm < stopSec
        if ok:
            ftm = round(cam.get(cv2.CAP_PROP_POS_MSEC) / 1000, 2)
            yield fno, ftm, img
            fno += 1


def rglob(*p):
    p = os.path.abspath('/**/'.join(p))
    ps = p.split('**')
    roots, ps = ps[0], ps[1:]
    if not ps:
        return glob(roots)
    else:
        ps = '**' + '**'.join(ps)
        res = []
        for root in glob(roots):
            for p in Path(root).glob(ps):
                res.append(str(p))
        return res


def getTraceBack(searchPys=None):
    errorTraceBooks = [basename(p) for p in searchPys or []]
    otrace = traceback.format_exc()
    trace = otrace.strip().split('\n')
    msg = trace[-1]
    done = False
    traces = [line.strip() for line in trace if line.strip().startswith('File "')]
    errLine = ''
    for line in traces[::-1]:
        if done:
            break
        meta = line.split(',')
        pyName = basename(meta[0].split(' ')[1].replace('"', ''))
        for book in errorTraceBooks:
            if book == pyName:
                done = True
                msg = f"{msg}, {' '.join(meta[1:])}. {meta[0]}"
                errLine = line
                break
    traces = '\n'.join(traces)
    traces = f"""
{msg}    


{otrace}


{traces}


{errLine}
"""
    return msg, traces


def filename(path, returnPath=False):
    if not returnPath:
        path = basename(path)
    return os.path.splitext(path)[0]
