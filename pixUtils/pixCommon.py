import os
import cv2
import sys
import shutil
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

np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, formatter={'float': lambda x: "{0:8.3f}".format(x)})


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


def str2path(*dirpath):
    dirpath = list(map(str, dirpath))
    path = join(*dirpath)
    if path.startswith('home/ec2-user'):
        path = join('/', path)
    return path


def moveCopy(src, des, op, isFile, rm):
    des = str2path(des)
    if isFile and not os.path.splitext(des)[-1]:
        raise Exception(f'''Fail des: {des}
                                    should be file''')
    if not rm and exists(des):
        raise Exception(f'''Fail des: {des}
                                    already exists delete it before operation''')
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


def dirop(*dirpath, **kw):
    mkdir, remove, mode = kw.get('mkdir', True), kw.get('rm'), kw.get('mode', 0o777)
    copyTo, moveTo = kw.get('cp'), kw.get('mv')

    assert kw.get('remove') is None
    assert kw.get('copyTo') is None
    assert kw.get('moveTo') is None

    path = str2path(*dirpath)
    isFile = os.path.splitext(path)[-1]
    if copyTo or moveTo:
        if not exists(path):
            raise Exception(f'''Fail src: {path}
                                            not found''')
    elif remove is True and exists(path):
        if isFile:
            os.remove(path)
        else:
            shutil.rmtree(path, ignore_errors=True)
    mkpath = dirname(path) if isFile else path
    if mkdir and not exists(mkpath) and mkpath:
        os.makedirs(mkpath)
    if copyTo:
        copy = shutil.copy if isFile else shutil.copytree
        path = moveCopy(path, copyTo, copy, isFile, rm=remove)
    elif moveTo:
        path = moveCopy(path, moveTo, shutil.move, isFile, rm=remove)
    return path


def getTimeStamp():
    return dt.now().strftime("%H_%M%b%d_%S%f")


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
