from abc import ABC

import torch
import random
from torch import nn
from pixUtils import *
import albumentations as A
import torch.nn.functional as F
from skimage.filters import gaussian
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, IterableDataset

torch.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)

bgColor = np.array([16, 196, 48], 'u1')
device = "cuda" if torch.cuda.is_available() else "cpu"


def changeBg(image, target, bgclr=bgColor, ignorePix=128):
    target = target.astype('f4').copy()
    if ignorePix is not None:
        target[target == ignorePix] = 0.5
    image = image[..., :3]
    if len(target.shape) == 2:
        target = target[..., None]
    target = image * target + bgclr * (1 - target)
    return target.astype(image.dtype)


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


def torch2mask(x, dtype='f4'):
    # nCh = len(x.shape)
    # if nCh == 4:
    #     x = torch.squeeze(x, dim=1)
    # if nCh == 3:
    # x = x[0]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        if dtype is not None:
            x = x.astype(dtype)
    return x


def weightFreeze(model, freezes=None, unfreezes=None, freezeBn=(), disp=False):
    if freezes is not None:
        if freezes == 'all':
            print("freezing all layers")
            for name, weights in model.named_parameters():
                weights.requires_grad = False
        else:
            for name, weights in model.named_parameters():
                for lname in freezes:
                    if lname in name:
                        print(f"freezing: {name}")
                        weights.requires_grad = False
    elif unfreezes is not None:
        if unfreezes == 'all':
            print("unfreezing all layers")
            for name, weights in model.named_parameters():
                weights.requires_grad = True
        else:
            for name, weights in model.named_parameters():
                for lname in unfreezes:
                    if lname in name:
                        print(f"unfreezing: {name}")
                        weights.requires_grad = True
    elif freezeBn:
        print(f"""freezing batch norm layers: {freezeBn}""")
        for weights in model.modules():
            for bnType in freezeBn:
                if isinstance(weights, bnType):
                    weights.eval()
                    # dir2(weights)
                    # weights.weight.requires_grad = False
                    # weights.bias.requires_grad = False
    else:
        print("_____________________________________________________________")
        for name, weights in model.named_parameters():
            print(f"{weights.requires_grad}\t\t{name}")
        print("_____________________________________________________________")
        for weights in model.modules():
            name = str(weights).replace('\n', '')
            print(f"{weights.training}\t\t{name}")


def loadWeights(model, weights, layerMap=None, debug=None, wkey=None):
    def weight_init(m):
        import torch.nn.init as init
        '''
        Usage:
            model = Model()
            model.apply(weight_init)
        '''
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            init.normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose3d):
            init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm3d):
            init.normal_(m.weight.data, mean=1, std=0.02)
            init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.LSTMCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRU):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)
        elif isinstance(m, nn.GRUCell):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    init.orthogonal_(param.data)
                else:
                    init.normal_(param.data)

    if not debug:
        weight_init(model)
    if type(weights) == str:
        print(f"loading weights from path: {weights} and wkey: {wkey}")
        weights = torch.load(weights)
        if wkey:
            weights = weights[wkey]
    elif type(weights) in [list, tuple]:
        print(f"downloading weights from torch hub: {weights}")
        weights = torch.hub.load(*weights).state_dict()
    if not isinstance(weights, OrderedDict):
        weights = weights.state_dict()
    if debug:
        print(pd.DataFrame([(k, v.shape, v.dtype) for k, v in weights.items()]))
    layerMap = layerMap or dict()
    newWeights = OrderedDict()
    for lname, lweight in weights.items():
        mapData = layerMap.get(lname)
        if mapData:
            if len(mapData) == 1:
                lname = mapData[0]  # name
            elif len(mapData) == 2:
                lname, lweight = mapData[0], lweight[mapData[1]]  # axis0
            elif len(mapData) == 3:
                lname, lweight = mapData[0], lweight[mapData[1]]
                lweight = lweight[:, mapData[2]]  # axis1
        newWeights[lname] = lweight
    if debug:
        print(pd.DataFrame([(k, v.shape, v.dtype) for k, v in newWeights.items()]))
    print("_________________________ loading weights _________________________")
    msg = model.load_state_dict(newWeights, strict=False)
    data = [f"'{d}'," for d in model.state_dict() if d not in msg.missing_keys]
    nData = len(data)
    if debug:
        data = '\n'.join(data)
        print(f"ok = [\n{data}\n]")
    data = '\n'.join([f"'{d}'," for d in msg.missing_keys])
    print(f"model = [\n{data}\n]")
    data = '\n'.join([f"'{d}'," for d in msg.unexpected_keys])
    print(f"weight = [\n{data}\n]")
    print("nWeightLoaded: ", nData)
    print("nModelMiss   : ", len(msg.missing_keys))
    print("nWeightMiss  : ", len(msg.unexpected_keys))
    if debug:
        print("195 loadWeights torchCommon : ", );
        quit()


def describeModel(model, x=(384, 384), batchSize=8, device=device, summary=True, fps=False, remove=(), *a, **kw):
    cols = ("Index", "Type", "Channels", "Kernel Shape", "Output Shape", "Params", "Mul Add")
    remove = [x.lower().replace(' ', '') for x in remove]
    cols = [x for x in cols if x.lower().replace(' ', '') not in remove]
    removeIndex = False if "Index" in cols else True
    if not removeIndex:
        cols.pop(0)
    x = np.array(x).round().astype(int)
    r, c = x

    def getSummary(model, x, *args, **kwargs):
        def get_names_dict(model):
            """Recursive walk to get names including path."""
            names, types = dict(), dict()

            def _get_names(module, parent_name=""):
                def decodeKey(key):
                    try:
                        int(key)
                        key = f"[{key}]"
                        # key = f".{key}"
                    except:
                        key = f".{key}"
                    return key

                for key, m in module.named_children():
                    cls_name = str(m.__class__).split(".")[-1].split("'")[0]
                    num_named_children = len(list(m.named_children()))
                    name = f"{parent_name}{decodeKey(key)}" if parent_name else key
                    names[name] = m
                    types[name] = cls_name

                    if isinstance(m, torch.nn.Module):
                        _get_names(m, parent_name=name)

            _get_names(model)
            return names, types

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

                    # ignore N, C when calculate Mul Add in ConvNd
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

        module_names, types = get_names_dict(model)

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
        types = [types['_'.join(name.split('_')[1:])] for name in df.index]
        df['Type'] = np.array(types).T
        df["Mul Add"] = pd.to_numeric(df["macs"], errors="coerce")
        df["Params"] = pd.to_numeric(df["params"], errors="coerce")
        df["Non-trainable params"] = pd.to_numeric(df["params_nt"], errors="coerce")
        inData = [list(x.shape)] + df.out.tolist()[:-1]
        df = df.rename(columns=dict(ksize="Kernel Shape", out="Output Shape"))
        df["Input Shape"] = inData
        df["Channels"] = df[["Input Shape", "Output Shape"]].applymap(lambda x: x[1]).to_numpy().tolist()
        df_sum = df.sum()
        df.index.name = "Layer"
        if removeIndex:
            df.index = [''] * len(df)
        df = df[list(cols)]

        option = pd.option_context("display.max_rows", None, "display.max_columns", 10, "display.float_format", pd.io.formats.format.EngFormatter(use_eng_prefix=True))
        with option:
            max_repr_width = max([len(row) for row in df.to_string().split("\n")])
            print("=" * max_repr_width)
            print(df.replace(np.nan, "-"))
            print("-" * max_repr_width)
            df_total = pd.DataFrame(
                    {"Total params": (df_sum["Params"] + df_sum["params_nt"]),
                     "Trainable params": df_sum["Params"],
                     "Non-trainable params": df_sum["params_nt"],
                     "Mul Add": df_sum["Mul Add"]
                     },
                    index=['Totals']
            ).T
            print(df_total)
            print("=" * max_repr_width)
        return df

    if device is not None:
        model = model.to(device)
    print("model", type(model))
    df = None
    if summary:
        __x = torch.zeros(2, next(model.parameters()).shape[1], r, c)
        if device is not None:
            __x = __x.to(device)
        df = getSummary(model, __x, *a, **kw)
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
        print(f"""
xShape: {__x.shape}
fps   : {fps:9.6f}
ms    : {1 / fps:9.6f}""")
    return df


def applyTransforms(transformers, data):
    for transformer in transformers:
        if type(transformer) == list:
            data = applyTransforms(transformer, data)
        else:
            data.update(transformer(**data))
    return data


def anyOneTransformer(transformers, p=.5):
    def getTransformer(**data):
        res = transformers[1]
        if random.random() < p:
            res = transformers[0]
        return applyTransforms([res], data)

    return getTransformer


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


class GenVideoData(IterableDataset, ABC):
    def __init__(self, yieldFn, transformers, nData, returnKeys):
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
    # alpha = np.float32(alpha != 0)
    if blend:
        alpha = gaussian(alpha, sigma=sigma, preserve_range=True)
    if len(img.shape) == 3:
        alpha = alpha[..., None]
    alpha *= alphaWeight
    img = paste_img * alpha + img * (1 - alpha)
    img = img.astype(img_dtype)
    return img


def CopyPaste(srcFn, pasteFn, minOverlay=.7, p=0.5):
    def __copyPaste(**data):
        src = applyTransforms(srcFn, data)
        image, mask = src['image'], src['mask']
        if random.random() < p:
            pdata = next(data.pop('pastePlayer'), None)
            if pdata:
                fno, ftm, vname, data['image'], data['mask'] = pdata
                paste = applyTransforms(pasteFn, data)
                pasteImage, pasteMask = paste['image'], paste['mask']
                f = pasteMask.astype('f4') / 255
                image = image_copy_paste(image, pasteImage, f, blend=False, alphaWeight=min(1.0, minOverlay + random.random()))
                mask = image_copy_paste(mask, pasteMask, f, blend=False, alphaWeight=1)
        data['image'], data['mask'] = image, mask
        return data

    return __copyPaste
