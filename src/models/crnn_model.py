# dans ce fichier, nous utilisons l'implémentation de CRNN faite dans le repoository github suivant : 
# https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py

# first we copy paste the CRNN model implementation from repository
# then we wrapp it in our OCRModel class


import torch.nn as nn
from src.models.base import OCRModel


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
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
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output




import torch
from PIL import Image


class CRNNModel(OCRModel):
    """Wrapper around CRNN model for handwritten text recognition."""
    
    def __init__(
        self,
        imgH: int = 32,                    # Hauteur de l'image (doit être multiple de 16)
        nc: int = 1,                       # Nombre de canaux (1=grayscale, 3=RGB)
        nclass: int = 37,                  # Nombre de classes (alphabet + blank pour CTC)
        nh: int = 256,                     # Nombre d'unités cachées dans LSTM
        leakyRelu: bool = False,           # Utiliser LeakyReLU au lieu de ReLU
        model_path: str | None = None,     # Chemin vers les poids pré-entraînés (optionnel)
        device: str | None = None,         # Device ('cpu' ou 'cuda')
        alphabet: str = "0123456789abcdefghijklmnopqrstuvwxyz", # Vocabulaire
    ) -> None:
        """
        Initialise le modèle CRNN.
        
        Args:
            imgH: Hauteur des images d'entrée (doit être multiple de 16)
            nc: Nombre de canaux (1 pour grayscale, 3 pour RGB)
            nclass: Nombre de classes de sortie (len(alphabet) + 1 pour CTC blank)
            nh: Taille des couches cachées du LSTM
            leakyRelu: Utiliser LeakyReLU au lieu de ReLU
            model_path: Chemin vers les poids pré-entraînés (si None, modèle non entraîné)
            device: Device PyTorch ('cpu' ou 'cuda')
            alphabet: Chaîne contenant tous les caractères possibles
        """
        self._imgH = imgH
        self._nc = nc
        self._nclass = nclass
        self._nh = nh
        self._leakyRelu = leakyRelu
        self._model_path = model_path
        self._alphabet = alphabet
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Vérification de cohérence
        if nclass != len(alphabet) + 1:
            # +1 pour le blank token du CTC
            import logging
            logging.warning(
                f"nclass ({nclass}) ne correspond pas à len(alphabet)+1 ({len(alphabet)+1}). "
                f"Ajustement automatique à {len(alphabet)+1}"
            )
            self._nclass = len(alphabet) + 1
        
        # Créer le modèle CRNN
        self._model = CRNN(
            imgH=self._imgH,
            nc=self._nc,
            nclass=self._nclass,
            nh=self._nh,
            n_rnn=2,  # Toujours 2 dans cette implémentation
            leakyRelu=self._leakyRelu
        )
        
        # Charger les poids pré-entraînés si fournis
        if model_path is not None:
            self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        
        self._model.to(self._device)
        self._model.eval()
        
        # Créer le dictionnaire de décodage pour CTC
        self._char_to_idx = {char: idx + 1 for idx, char in enumerate(self._alphabet)}
        self._idx_to_char = {idx + 1: char for idx, char in enumerate(self._alphabet)}
        self._idx_to_char[0] = ""  # Blank token pour CTC
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Prétraite l'image pour le modèle CRNN.
        
        Args:
            image: Image PIL
            
        Returns:
            Tensor PyTorch de forme (1, nc, imgH, imgW)
        """
        # Convertir en grayscale si nécessaire
        if self._nc == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        
        # Redimensionner en gardant le ratio d'aspect
        w, h = image.size
        ratio = w / float(h)
        imgW = int(self._imgH * ratio)
        
        # Redimensionner l'image
        image = image.resize((imgW, self._imgH), Image.BILINEAR)
        
        # Convertir en tensor et normaliser [0, 1]
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * self._nc, (0.5,) * self._nc)  # Normaliser [-1, 1]
        ])
        
        tensor = transform(image)
        tensor = tensor.unsqueeze(0)  # Ajouter dimension batch
        
        return tensor
    
    def _decode_predictions(self, preds: torch.Tensor) -> str:
        """
        Décode les prédictions du modèle en texte.
        
        Args:
            preds: Prédictions du modèle de forme (seq_len, batch, nclass)
            
        Returns:
            Texte décodé
        """
        # Prendre l'argmax pour obtenir les indices de classes
        _, preds = preds.max(2)  # (seq_len, batch)
        preds = preds.squeeze(1)  # (seq_len,)
        
        # Décoder avec CTC (supprimer les répétitions et les blanks)
        decoded_text = []
        prev_char = None
        
        for idx in preds:
            idx = idx.item()
            
            # Ignorer le blank token (0)
            if idx == 0:
                prev_char = None
                continue
            
            # Ignorer les répétitions (règle CTC)
            char = self._idx_to_char.get(idx, "")
            if char != prev_char:
                decoded_text.append(char)
                prev_char = char
        
        return "".join(decoded_text)
    
    def predict(self, image: ImageInput) -> str:
        """
        Effectue la prédiction OCR sur l'image fournie.
        
        Args:
            image: Chemin vers l'image ou objet PIL.Image
            
        Returns:
            Texte prédit
        """
        # Charger l'image si c'est un chemin
        pil_image = Image.open(image) if isinstance(image, str) else image
        
        # Prétraiter l'image
        tensor = self._preprocess_image(pil_image).to(self._device)
        
        # Prédiction
        with torch.no_grad():
            preds = self._model(tensor)  # (seq_len, batch, nclass)
        
        # Décoder les prédictions
        text = self._decode_predictions(preds)
        
        return text
    
    def get_name(self) -> str:
        """Retourne le nom du modèle."""
        model_info = f"CRNN(imgH={self._imgH}, nc={self._nc}, nh={self._nh}"
        if self._model_path:
            model_info += f", pretrained={self._model_path}"
        model_info += ")"
        return model_info