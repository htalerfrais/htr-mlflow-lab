# dans ce fichier, nous utilisons l'implémentation de CRNN faite dans le repository github suivant : 
# https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py

# d'abord on copie/colle les 4 classes depuis ce repository
# ensuite on les wrap dans notre classe OCRModel

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import collections  # necessaire pour strLabelConverter

from .base import OCRModel, ImageInput



# ===== CLASS 1 & 2: Model classes =====
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        
        cnn = nn.Sequential()
        
        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        convRelu(6, True)
        
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
    
    def forward(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output


# ===== CLASS 3: Label converter =====
class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'
        
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1
    
    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.abc.Iterable):  # Fixed for Python 3.9+
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
    
    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum()
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts


# ===== CLASS 4: Image preprocessor =====
class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
    
    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


# ===== CLASS 5: OCRModel wrapper =====

class CRNNModel(OCRModel):
    """CRNN-based OCR model implementation."""
    
    def __init__(
        self,
        model_path: str = "models_local/crnn.pth",
        alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyz",
        img_height: int = 32,
        img_width: int = 100,
        n_hidden: int = 256,
        n_channels: int = 1,
        device: str = 'cpu'
    ):
        """
        Initialize a CRNN model for OCR.
        
        Args:
            model_path: Path to the trained model weights (.pth file)
            alphabet: String of all characters the model can recognize
            img_height: Height of input images (must be multiple of 16)
            img_width: Width of input images
            n_hidden: Number of hidden units in LSTM
            n_channels: Number of input channels (1 for grayscale, 3 for RGB)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.img_height = img_height
        self.img_width = img_width
        
        # Initialize components
        n_class = len(alphabet) + 1  # +1 for blank token
        self.model = CRNN(img_height, n_channels, n_class, n_hidden)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()
        
        self.converter = strLabelConverter(alphabet)
        # Model-specific preprocessing: resizeNormalize adapts image dimensions
        # to match the CRNN architecture requirements (e.g., height must be multiple of 16)
        # This is separate from common preprocessing applied before model.predict()
        self.transform = resizeNormalize((img_width, img_height))
    
    def predict(self, image: ImageInput) -> str:
        """
        Run OCR inference on an image.
        
        Note: This method applies model-specific preprocessing (resizeNormalize)
        to adapt the image dimensions to the CRNN architecture. Common preprocessing
        (binarization, cropping, etc.) should be applied before calling this method
        via the pipeline's preprocessor.
        
        Args:
            image: Either a path to an image file or a PIL Image object
                  (may already be preprocessed by the common preprocessor)
            
        Returns:
            str: The recognized text from the image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert('L')  # Convert to grayscale
        elif image.mode != 'L':
            image = image.convert('L')
        
        # Model-specific preprocessing: adapt dimensions to CRNN architecture
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            preds = self.model(image_tensor)
        
        # Decode predictions
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = torch.IntTensor([preds.size(0)])
        raw_pred = self.converter.decode(preds.data, preds_size, raw=False)
        
        return raw_pred
    
    def get_name(self) -> str:
        """Return a human-readable model name."""
        return "CRNN"