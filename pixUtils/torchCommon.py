from pixUtils import *
import torch
import torch.nn.functional as F
from skimage.filters import gaussian
from torchvision.transforms import Normalize
from torch.utils.data import Dataset, DataLoader, IterableDataset

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


def normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), reverse=False, maxPix=255):
    """
      the albumentations results are not matching with this implementation
      need to test more with albumentations [maxPix=255]
    """

    def _normalize(image, mask=None, mean=mean, std=std, reverse=reverse):
        if reverse:
            std = 1 / np.array(std)
            mean = -np.array(mean) * std
        print(mean, std)
        image -= np.array(mean)
        image /= np.array(std)
        return dict(image=image, mask=mask)

    return _normalize


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
    x = torch.unsqueeze(x, dim=1) if len(x.shape) == 3 else x[None]
    return x.to(device)


def torch2img(x):
    x = x.permute(1, 2, 0) if len(x.shape) == 3 else x.permute(0, 2, 3, 1)
    return x.cpu().numpy()


def torch2mask(x):
    x = torch.squeeze(x, dim=1) if len(x.shape) == 4 else x[0]
    return x.cpu().numpy()


def genGuideBook(dataRootPaths, desDir, nSamples=None):
    assert type(dataRootPaths) == list
    valDatasPath = f'{desDir}/valPair.txt'
    trainDatasPath = f'{desDir}/trainPair.txt'

    imPat = '{data}/JPEGImages/{iname}.jpg'
    segPat = '{data}/SegmentationClassRaw/{iname}.png'
    valBook = '{data}/ImageSets/Segmentation/val.txt'
    trainBook = '{data}/ImageSets/Segmentation/train.txt'

    for guideBook, desPath in [(trainBook, trainDatasPath), (valBook, valDatasPath)]:
        with open(desPath, 'w') as book:
            for data in dataRootPaths:
                lines = Path(guideBook.format(data=data)).read_text().split('\n')
                if nSamples is not None:
                    lines = lines[:nSamples]
                for iname in tqdm(lines, desc=f"{basename(data)} [{basename(guideBook)}]"):
                    iname = iname.strip()
                    if iname:
                        line = f"{basename(data)}, {imPat.format(data=data, iname=iname)}, {segPat.format(data=data, iname=iname)}\n"
                        book.write(line)

    for dataPath in [trainDatasPath, valDatasPath]:
        print(f'\n____________________________{dataPath}____________________________')
        data = Path(dataPath).read_text().split('\n')
        print("len(data)", len(data))
        for mask in data[:5]:
            print(mask)
    print("\n______________________________________________________________________________")
    return trainDatasPath, valDatasPath


def applyTransforms(image, mask, transformers, hasLabels):
    if hasLabels:
        for transformer in transformers:
            data = transformer(image=image, mask=mask)
            image, mask = data['image'], data['mask']
    else:
        for transformer in transformers:
            image = transformer(image=image)['image']
    return image, mask


class DummyDataLoader:
    def __init__(self, dataset, *a, **kw):
        self.dataset = dataset

    def __iter__(self):
        for odata, timage, tmask in self.dataset:
            odata = [odata]
            yield odata, timage[None, ...], tmask[None, ...]


def readBook(guidePath):
    with open(guidePath, 'r') as book:
        lines = book.read().strip().split('\n')
    return lines


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    img_dtype = img.dtype
    if len(alpha.shape) == 3:
        alpha = alpha.max(axis=-1)
    alpha = np.float32(alpha != 0)
    if blend:
        alpha = gaussian(alpha, sigma=sigma, preserve_range=True)
    if len(img.shape) == 3:
        alpha = alpha[..., None]
    img = paste_img * alpha + img * (1 - alpha)
    img = img.astype(img_dtype)
    return img


def cropFace(image, mask, bbox, landMarks, p=0.5):
    if bbox is not None and np.random.rand(1)[0] < p:
        bbox = bboxScale(image, bbox, 2)
        image = getSubImg(image, bbox)
        mask = getSubImg(mask, bbox)
    return image, mask, bbox, landMarks


class GenImgData(Dataset):
    def __init__(self, guidePath, transformers, postTransformers):
        self.transformers = transformers
        self.postTransformers = postTransformers
        self.dataPaths = [line.split(', ') for line in readBook(guidePath)]

    def __len__(self):
        return len(self.dataPaths)

    def postPrcs(self, image, mask):
        image, mask = applyTransforms(image, mask, self.postTransformers, True)
        mask = torch.Tensor() if mask is None else mask
        return image, mask

    def getData(self, ix):
        dtype, image, mask = self.dataPaths[ix]
        mask = cv2.imread(mask)
        image = cv2.imread(image)
        return dtype, image, mask

    def __getitem__(self, ix):
        dtype, image, mask = self.getData(ix)
        image, mask = applyTransforms(image, mask, self.transformers, True)
        image, mask = self.postPrcs(image, mask)
        return ix, image, mask


class GenImgDataCopyPasteAug(Dataset):
    def __init__(self, guidePath, faceBoxPath, transformers, pasteTransformers, postTransformers):
        self.transformers = transformers
        self.postTransformers = postTransformers
        self.pasteTransformers = pasteTransformers
        self.dataPaths = [line.split(', ') for line in readBook(guidePath)]
        self.faceBoxs = [json.loads(d) for d in readBook(faceBoxPath) if d.strip()]
        self.faceBoxs = {d['impath']: d for d in self.faceBoxs}

    def __len__(self):
        return len(self.dataPaths)

    def postPrcs(self, image, mask):
        image, mask = applyTransforms(image, mask, self.postTransformers, True)
        mask = torch.Tensor() if mask is None else mask
        return image, mask

    def getData(self, ix):
        dtype, impath, maskpath = self.dataPaths[ix]
        faceData = self.faceBoxs.get(impath, dict(impath=impath, rect=None, landMarks=None))
        mask = cv2.imread(maskpath)
        image = cv2.imread(impath)
        return dtype, image, mask, faceData['rect'], faceData['landMarks']

    def __getitem__(self, ix):
        # TODO integrate faceCrop copyPaste inside albumentations
        dtype, image, mask, bbox, landMarks = self.getData(ix)
        image, mask, bbox, landMarks = cropFace(image, mask, bbox, landMarks)
        image, mask = applyTransforms(image, mask, self.transformers, True)

        if np.random.rand(1)[0] < 0.5:
            for i in range(np.random.randint(1, 3)):
                dtype, pasteImage, pasteMask, pasteBbox, pasteLandMarks = self.getData(np.random.randint(len(self.dataPaths)))
                # pasteImage, pasteMask, pasteBbox, pasteLandMarks = self.cropFace(pasteImage, pasteMask, pasteBbox, pasteLandMarks)
                pasteImage, pasteMask = applyTransforms(pasteImage, pasteMask, self.pasteTransformers, True)

                image = image_copy_paste(image, pasteImage, pasteMask, blend=True)
                mask = image_copy_paste(mask, pasteMask, pasteMask, blend=False)
        image, mask = self.postPrcs(image, mask)
        return ix, image, mask


class GenVideoData(IterableDataset):
    def __init__(self, vpaths, transformers, skipFrame=10):
        self.vpaths = vpaths
        self.skipFrame = skipFrame
        self.transformers = transformers

    @staticmethod
    def postPrcs(fno, image, mask):
        return fno, image, mask

    def __iter__(self):
        for vpath in self.vpaths:
            print("processing: ", vpath)
            for ix, (fno, ftm, image) in enumerate(videoPlayer(vpath)):
                mask = None
                if ix % self.skipFrame == 0:
                    image, mask = applyTransforms(image, mask, self.transformers, False)
                    mask = torch.Tensor() if mask is None else mask
                    yield self.postPrcs(fno, image, mask)


def inferDatas(model, testDatas, device):
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (odatas, xs, ys) in tqdm(enumerate(testDatas)):
            tik = clk()
            ps = model(xs.to(device, non_blocking=True))
            tok = tik.tok("").last()
            ps = torch.argmax(ps, dim=1)
            ps = ps.cpu().numpy()
            for (ix, ox, oy), p in zip(zip(*odatas), ps):
                ox, oy = ox.numpy(), oy.numpy()
                yield ix, tok, ox, oy, p
