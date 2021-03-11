import torch
import random
from torch import nn
from pixUtils import *
import albumentations as A
import torch.nn.functional as F
from skimage.filters import gaussian
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, IterableDataset

bgColor = np.array([16, 196, 48], 'u1')
device = "cuda" if torch.cuda.is_available() else "cpu"


def changeBg(image, target, bgclr):
    image = image[..., :3]
    if len(target.shape) == 2:
        target = target[..., None]
    target = image * target + bgclr * (1 - target)
    return target.astype(image.dtype)


def torchChangeBG(images, outputs, bgColor):
    res = []
    for image, output in zip(images, outputs):
        image = image[:3, ...]
        output = output[None, ...]
        output = image * output + bgColor * (1 - output)
        res.append(output)
    return res


def Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0, reverse=False):
    if reverse:
        stdR = np.reciprocal(std)
        meanR = -np.array(mean) * stdR
        return A.Normalize(mean=meanR, std=stdR, max_pixel_value=1.0, always_apply=always_apply, p=p)
    else:
        return A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=always_apply, p=p)


def img2torch(x, device):
    if type(x) == list:
        x = np.array(x)
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1) if len(x.shape) == 3 else x.permute(0, 3, 1, 2)
    return x.to(device)


def mask2torch(x, device):
    if type(x) == list:
        x = np.array(x)
    x = torch.from_numpy(x)
    nCh = len(x.shape)
    if nCh == 3:
        x = torch.unsqueeze(x, dim=1)
    else:
        x = x[None]
    return x.to(device)


def torch2img(x, normalize=Normalize(reverse=True), float2img=True):
    nCh = len(x.shape)
    if nCh == 4:
        x = x.permute(0, 2, 3, 1)
    if nCh == 3:
        x = x.permute(1, 2, 0)
    x = x.detach().cpu().numpy()
    if normalize:
        x = normalize(image=x)['image']
    if float2img:
        x = np.uint8(255 * x)
    return x


def torch2mask(x):
    nCh = len(x.shape)
    # if nCh == 4:
    #     x = torch.squeeze(x, dim=1)
    # if nCh == 3:
    # x = x[0]
    return x.detach().cpu().numpy()


def loadWeights(model, weights, layerNameMap=None, disp=None, dispAll=True):
    layerNameMap = layerNameMap or {}
    if type(weights) == str:
        weights = torch.load(weights)
    if disp:
        if not isinstance(disp, OrderedDict):
            disp = disp.state_dict()
        for k, v in disp.items():
            if dispAll:
                print(k, v.shape, v.dtype)
            else:
                print(k)
        return
    newWeights = OrderedDict()
    for lname, lweight in weights.items():
        newName = layerNameMap.get(lname)
        if newName:
            lname = newName
        newWeights[lname] = lweight
    print("_________________________loading weights_________________________")
    msg = model.load_state_dict(newWeights, strict=False)
    data = [f"'{d}'," for d in model.state_dict() if d not in msg.missing_keys]
    nData = len(data)
    data = '\n'.join(data)
    print(f"ok = [\n{data}\n]")
    data = '\n'.join([f"'{d}'," for d in msg.missing_keys])
    print(f"model = [\n{data}\n]")
    data = '\n'.join([f"'{d}'," for d in msg.unexpected_keys])
    print(f"weight = [\n{data}\n]")
    print("len(ok)", nData)
    print("len(model)", len(msg.missing_keys))
    print("len(weight)", len(msg.unexpected_keys))


def describeModel(model, x=(384, 384), batchSize=8, device=device, summary=True, fps=False,
        cols=("Kernel Shape", "Input Shape", "Output Shape", "Params", "Mult-Adds"),
        removeIndex=False, *a, **kw):
    r, c = x

    def getSummary(model, x, *args, **kwargs):
        def get_names_dict(model):
            """Recursive walk to get names including path."""
            names = {}

            def _get_names(module, parent_name=""):
                for key, m in module.named_children():
                    cls_name = str(m.__class__).split(".")[-1].split("'")[0]
                    num_named_children = len(list(m.named_children()))
                    if num_named_children > 0:
                        name = parent_name + "." + key if parent_name else key
                    else:
                        name = parent_name + "." + cls_name + "_" + key if parent_name else key
                    names[name] = m

                    if isinstance(m, torch.nn.Module):
                        _get_names(m, parent_name=name)

            _get_names(model)
            return names

        def register_hook(module):
            # ignore Sequential and ModuleList
            if not module._modules:
                hooks.append(module.register_forward_hook(hook))

        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            # Lookup name in a dict that includes parents
            for name, item in module_names.items():
                if item == module:
                    key = "{}_{}".format(module_idx, name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                try:
                    info["out"] = list(outputs[0].size())
                except AttributeError:
                    # pack_padded_seq and pad_packed_seq store feature into data attribute
                    info["out"] = list(outputs[0].data.size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["inner"] = OrderedDict()
            info["params_nt"], info["params"], info["macs"] = 0, 0, 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement() * param.requires_grad
                info["params_nt"] += param.nelement() * (not param.requires_grad)

                if name == "weight":
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["macs"] = "-", "-"

            summary[key] = info

        module_names = get_names_dict(model)

        hooks = []
        summary = OrderedDict()

        model.apply(register_hook)
        try:
            with torch.no_grad():
                # model([x], [[dict(ori_shape=[384, 384], img_shape=[384, 384], pad_shape=[384, 384], flip=False)]], {}) # TODO vishnu
                model(x) if not (kwargs or args) else model(x, *args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()

        # Use pandas to align the columns
        df = pd.DataFrame(summary).T

        df["Mult-Adds"] = pd.to_numeric(df["macs"], errors="coerce")
        df["Params"] = pd.to_numeric(df["params"], errors="coerce")
        df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
        inData = [list(x.shape)] + df.out.tolist()[:-1]
        df = df.rename(columns=dict(ksize="Kernel Shape", out="Output Shape"))
        df["Input Shape"] = inData
        df_sum = df.sum()
        df.index.name = "Layer"
        if removeIndex:
            df.index = [''] * len(df)
        df = df[list(cols)]
        option = pd.option_context("display.max_rows", 600, "display.max_columns", 10, "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True))
        with option:
            max_repr_width = max([len(row) for row in df.to_string().split("\n")])
            print("=" * max_repr_width)
            print(df.replace(np.nan, "-"))
            print("-" * max_repr_width)
            df_total = pd.DataFrame(
                    {"Total params": (df_sum["Params"] + df_sum["params_nt"]),
                     "Trainable params": df_sum["Params"],
                     "Non-trainable params": df_sum["params_nt"],
                     "Mult-Adds": df_sum["Mult-Adds"]
                     },
                    index=['Totals']
            ).T
            print(df_total)
            print("=" * max_repr_width)

    if device is not None:
        model = model.to(device)
    print("model", type(model))
    if summary:
        __x = torch.zeros(2, next(model.parameters()).shape[1], r, c)
        if device is not None:
            __x = __x.to(device)
        getSummary(model, __x, *a, **kw)
    if fps:
        __x = torch.zeros(batchSize, next(model.parameters()).shape[1], r, c)
        if device is not None:
            __x = __x.to(device)
        mm = model
        for i in range(5):
            mm(__x)
        nIter = 15
        tik = clk()
        for i in range(nIter):
            mm(__x)
        tok = tik.tok("").last()
        nFrame = nIter * __x.shape[0]
        fps = nFrame / tok
        print(f"fps: {fps:9.6f}\nms : {1/fps:9.6f}")


def genGuideBook(dataRootPaths, desDir, faceBoxs, nSamples=None):
    assert type(dataRootPaths) == list
    valDatasPath = f'{desDir}/valPair.txt'
    trainDatasPath = f'{desDir}/trainPair.txt'

    imPat = '{data}/JPEGImages/{iname}.jpg'
    segPat = '{data}/SegmentationClassRaw/{iname}.png'
    valBook = '{data}/ImageSets/Segmentation/val.txt'
    trainBook = '{data}/ImageSets/Segmentation/train.txt'
    startIx = []
    for guideBook, desPath in [(trainBook, trainDatasPath), (valBook, valDatasPath)]:
        counter = 0
        with open(desPath, 'w') as book:
            for data in dataRootPaths:
                startIx.append(counter)
                lines = Path(guideBook.format(data=data)).read_text().split('\n')
                if nSamples is not None:
                    lines = lines[:nSamples]
                for iname in tqdm(lines, desc=f"{basename(data)} [{basename(guideBook)}]"):
                    iname = iname.strip()
                    if iname:
                        jData = dict()
                        jData['dataset'] = basename(data)
                        impath = imPat.format(data=data, iname=iname)
                        jData['imPath'] = impath
                        jData['segPath'] = segPat.format(data=data, iname=iname)

                        jData.update(faceBoxs.get(impath, dict()))
                        book.write(f"{json.dumps(jData)}\n")
                        counter += 1

    for dataPath in [trainDatasPath, valDatasPath]:
        print(f'\n____________________________{dataPath}____________________________')
        data = Path(dataPath).read_text().split('\n')
        print("len(data)", len(data))
        for mask in data[:5]:
            print(mask)
    print("\n______________________________________________________________________________")
    return trainDatasPath, valDatasPath, startIx


def applyTransforms(transformers, data):
    for transformer in transformers:
        if type(transformer) == list:
            data = applyTransforms(transformer, data)
        else:
            data.update(transformer(**data))
    return data


class GenImgData(Dataset):
    def __init__(self, transformers, nData, returnKeys):
        self.nData = nData
        self.transformers = transformers
        self.returnKeys = returnKeys

    def __len__(self):
        return self.nData

    def __getitem__(self, ix):
        data = applyTransforms(self.transformers, data=dict(ix=ix))
        return {k: data[k] for k in self.returnKeys}


class GenVideoData(IterableDataset):
    def __init__(self, yieldFn, transformers, returnKeys):
        self.yieldFn = yieldFn
        self.transformers = transformers
        self.returnKeys = returnKeys

    def __iter__(self):
        for data in self.yieldFn():
            data = applyTransforms(self.transformers, data=data)
            yield {k: data[k] for k in self.returnKeys}


def GuideCrop(bbox_scale=2, p=.5):
    def __guideCrop(image, mask, bbox, **data):
        if bbox is not None and random.random() < p:
            bbox = bboxScale(image, bbox, bbox_scale)
            image = getSubImg(image, bbox)
            mask = getSubImg(mask, bbox)
        return dict(image=image, mask=mask)

    return __guideCrop


def image_copy_paste(img, paste_img, alpha, alphaWeight, blend=True, sigma=1):
    img_dtype = img.dtype
    if len(alpha.shape) == 3:
        alpha = alpha.max(axis=-1)
    alpha = np.float32(alpha != 0)
    if blend:
        alpha = gaussian(alpha, sigma=sigma, preserve_range=True)
    if len(img.shape) == 3:
        alpha = alpha[..., None]
    alpha *= alphaWeight
    img = paste_img * alpha + img * (1 - alpha)
    img = img.astype(img_dtype)
    return img


def CopyPaste(srcFn, pasteFn, p=0.5):
    def __copyPaste(**data):
        src = applyTransforms(srcFn, data)
        image, mask = src['image'], src['mask']
        if random.random() < p:
            data['image'], data['mask'], data['bbox'] = data.pop('pasteImg'), data.pop('pasteMask'), data.pop('pasteBox')
            paste = applyTransforms(pasteFn, data)
            pasteImage, pasteMask = paste['image'], paste['mask']
            image = image_copy_paste(image, pasteImage, pasteMask, blend=True, alphaWeight=min(1, .35 + np.random.random(1)[0]))
            mask = image_copy_paste(mask, pasteMask, pasteMask, blend=False, alphaWeight=1)
        data['image'], data['mask'] = image, mask
        return data

    return __copyPaste

# class GenImgData(Dataset):
#     def __init__(self, guidePath, nSamples, transformers, postTransformers):
#         self.transformers = transformers
#         self.postTransformers = postTransformers
#         self.dataPaths = [line.split(', ') for line in readBook(guidePath)]
#         if nSamples is not None:
#             self.dataPaths = self.dataPaths[:nSamples]
#
#     def __len__(self):
#         return len(self.dataPaths)
#
#     def postPrcs(self, image, mask):
#         image, mask = applyTransforms(image, mask, self.postTransformers, True)
#         mask = torch.Tensor() if mask is None else mask
#         return image, mask
#
#     def getData(self, ix):
#         dtype, image, mask = self.dataPaths[ix]
#         mask = cv2.imread(mask)
#         image = cv2.imread(image)
#         return dtype, image, mask
#
#     def __getitem__(self, ix):
#         dtype, image, mask = self.getData(ix)
#         image, mask = applyTransforms(image, mask, self.transformers, True)
#         image, mask = self.postPrcs(image, mask)
#         return ix, image, mask
#         return image, mask

# class GenImgDataCopyPasteAug(Dataset):
#     def __init__(self, guidePath, faceBoxPath, nSamples, transformers, pasteTransformers, postTransformers):
#         self.transformers = transformers
#         self.postTransformers = postTransformers
#         self.pasteTransformers = pasteTransformers
#         self.dataPaths = [line.split(', ') for line in readBook(guidePath)]
#         if nSamples is not None:
#             self.dataPaths = self.dataPaths[:nSamples]
#         self.faceBoxs = [json.loads(d) for d in readBook(faceBoxPath) if d.strip()]
#         self.faceBoxs = {d['impath']: d for d in self.faceBoxs}
#
#     def __len__(self):
#         return len(self.dataPaths)
#
#     def postPrcs(self, image, mask):
#         image, mask = applyTransforms(image, mask, self.postTransformers, True)
#         mask = torch.Tensor() if mask is None else mask
#         return image, mask
#
#     def getData(self, ix):
#         dtype, impath, maskpath = self.dataPaths[ix]
#         faceData = self.faceBoxs.get(impath, dict(impath=impath, rect=None, landMarks=None))
#         mask = cv2.imread(maskpath)
#         image = cv2.imread(impath)
#         return dtype, image, mask, faceData['rect'], faceData['landMarks']
#
#     def __getitem__(self, ix):
#         # TODO integrate faceCrop copyPaste inside albumentations
#         dtype, image, mask, bbox, landMarks = self.getData(ix)
#         image, mask, bbox, landMarks = cropFace(image, mask, bbox, landMarks)
#         image, mask = applyTransforms(image, mask, self.transformers, True)
#
#         if np.random.rand(1)[0] < 0.5:
#             for i in range(np.random.randint(1, 3)):
#                 dtype, pasteImage, pasteMask, pasteBbox, pasteLandMarks = self.getData(np.random.randint(len(self.dataPaths)))
#                 # pasteImage, pasteMask, pasteBbox, pasteLandMarks = self.cropFace(pasteImage, pasteMask, pasteBbox, pasteLandMarks)
#                 pasteImage, pasteMask = applyTransforms(pasteImage, pasteMask, self.pasteTransformers, True)
#
#                 image = image_copy_paste(image, pasteImage, pasteMask, blend=True)
#                 mask = image_copy_paste(mask, pasteMask, pasteMask, blend=False)
#         image, mask = self.postPrcs(image, mask)
#         return image, mask


# class GenVideoData(IterableDataset):
#     def __init__(self, vpaths, transformers, postTransformers, skipFrame=10):
#         self.vpaths = vpaths
#         self.skipFrame = skipFrame
#         self.transformers = transformers
#         self.postTransformers = postTransformers
#
#     def postPrcs(self, image, mask):
#         image, mask = applyTransforms(image, mask, self.postTransformers, True)
#         mask = torch.Tensor() if mask is None else mask
#         return image, mask
#
#     def __iter__(self):
#         for vpath in self.vpaths:
#             print("processing: ", vpath)
#             for ix, (fno, ftm, image) in enumerate(videoPlayer(vpath)):
#                 mask = None
#                 if ix % self.skipFrame == 0:
#                     image, mask = applyTransforms(image, mask, self.transformers, False)
#                     image, mask = self.postPrcs(image, mask)
#                     yield fno, image, mask


# def inferDatas(model, testDatas, device):
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for i, (odatas, xs, ys) in tqdm(enumerate(testDatas)):
#             tik = clk()
#             ps = model(xs.to(device, non_blocking=True))
#             tok = tik.tok("").last()
#             ps = torch.argmax(ps, dim=1)
#             ps = ps.cpu().numpy()
#             for (ix, ox, oy), p in zip(zip(*odatas), ps):
#                 ox, oy = ox.numpy(), oy.numpy()
#                 yield ix, tok, ox, oy, p


# def inferDatas(model, testDatas, device):
#     model.to(device)
#     model.eval()
#     with torch.no_grad():
#         for i, (ix, xs, ys) in tqdm(enumerate(testDatas)):
#             tik = clk()
#             ps = model(xs.to(device, non_blocking=True))
#             tok = tik.tok("").last()
#             ps = torch.argmax(ps, dim=1)
#             ps = ps.cpu().numpy()
#             for (ix, ox, oy), p in zip(zip(*odatas), ps):
#                 ox, oy = ox.numpy(), oy.numpy()
#                 yield ix, tok, ox, oy, p


# def getTransformers(imSize):
#     inSize = np.array(imSize, int)
#     border_mode = cv2.BORDER_CONSTANT
#     pasteSize = list(np.int32(.8 * inSize))
#     # border_mode = cv2.BORDER_REFLECT_101
#     trainReader, nTrain = ReadGuide(impaths, None, returnPaste=True)
#     inferReader, nInfer = ReadGuide(impaths, None, returnPaste=False)
#     srcT, pasteT, postT, inferT = list(), list(), list(), list()
#
#     srcT.append(prePrcs)
#     srcT.append(GuideCrop())
#     srcT.append(A.ShiftScaleRotate(shift_limit=0.15, scale_limit=(-0.1, .5), rotate_limit=15, border_mode=border_mode, p=1))
#     srcT.append(A.OneOf([
#         A.RandomResizedCrop(*inSize),
#         A.Resize(*inSize),
#     ], p=1))
#     srcT.append(A.HorizontalFlip())
#     srcT.append(A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5))
#     srcT.append(A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5))
#
#     pasteT.append(prePrcs)
#     pasteT.append(GuideCrop())
#     pasteT.append(A.ShiftScaleRotate(shift_limit=0.4, scale_limit=(-0.9, 1), rotate_limit=30, border_mode=border_mode, p=1))
#     pasteT.append(A.OneOf([
#         A.RandomResizedCrop(*pasteSize),
#         A.Resize(*pasteSize),
#     ], p=1))
#     pasteT.append(A.PadIfNeeded(*inSize, border_mode=border_mode))
#     pasteT.append(A.HorizontalFlip())
#     pasteT.append(A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5))
#     pasteT.append(A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5))
#
#     postT.append(A.ShiftScaleRotate(shift_limit=0.1, scale_limit=(-0.9, 1), rotate_limit=10, border_mode=border_mode, p=1))
#     postT.append(A.PadIfNeeded(*inSize, border_mode=border_mode))
#     postT.append(postPrcs)
#
#     inferT.append(prePrcs)
#     inferT.append(A.Resize(*inSize))
#     inferT.append(postPrcs)
#     # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#     # ToTensorV2(),
#     # return GenImgData([inferReader, inferTransformers], nInfer)
#     return GenImgData([trainReader, CopyPaste(srcT, pasteT), postT], nTrain)
