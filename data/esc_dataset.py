import torch
import os
import logging
import glob
import soundfile as sf
import numpy as np
import resampy
import sys
import torchaudio
from data.data_transformer import FakePitchShift, Compose
from data import register_dataset
torchaudio.set_audio_backend("soundfile")  # switch backend

logger = logging.getLogger(__name__)


@register_dataset
class EscRawAudio(torch.utils.data.Dataset):
    """
    sr=44100, 5seconds, mono.
    """
    def __init__(self, fold, split, target_sr=44100, root='/data/songhongwei/ESC-50/audio', transform=None):
        super(EscRawAudio, self).__init__()
        """
        split: train, valid, full
        fold: 1 to 5
        """
        if not os.path.exists(root):
            raise Exception("{} does not exists.".format(root))

        feat_files = glob.glob(os.path.join(root, "*.wav"))
        feat_set = set(feat_files)
        val_files = glob.glob(os.path.join(root, "{}*.wav".format(str(fold))))
        val_set = set(val_files)
        train_set = feat_set - val_set

        if split == "train":
            self.files = list(train_set)
        elif split == "valid":
            self.files = list(val_set)
        elif split == "full":
            self.files = list(feat_set)
        else:
            raise ValueError("split {} not supported.".format(self.split))

        _, sr = sf.read(file=self.files[0])
        if target_sr:
            self.target_sr = target_sr
        else:
            self.target_sr = sr
        logger.info("Original Sample Rate: {}, Target Sample Rate: {}".format(sr, target_sr))

        self.transform = transform
        logger.info("Loading fold {}, split {}, {} files".format(str(fold), split, len(self.files)))

    def __getitem__(self, idx):
        """
        :param idx:
        :return: audio, label
        """
        file = self.files[idx]
        if isinstance(self.transform, FakePitchShift):
            file = self.transform(file)
        if isinstance(self.transform, Compose):
            if isinstance(self.transform.transforms[0], FakePitchShift):
                file = self.transform.transforms[0](file)
        # (channels, frames)
        audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
        audio = self._process_audio(audio, sr, self.target_sr)
        label = int(file[-5:-4]) if file[-6:-4].startswith('-') else int(file[-6:-4])
        sample = (torch.as_tensor(audio, dtype=torch.float32), torch.as_tensor(label, dtype=torch.int64))
        if self.transform and not isinstance(self.transform, FakePitchShift):
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _process_audio(audio, sr, target_sr):
        if audio.size(0) == 2:
            # Downmix if multichannel
            audio = torch.mean(audio, dim=0, keepdim=True)

        audio = audio[0]

        if sr != target_sr:
            audio = resampy.resample(audio.numpy(), sr_orig=sr, sr_new=target_sr, filter='kaiser_best')

        return audio


if __name__ == '__main__':
    ds = EscRawAudio(fold=1, split='valid', target_sr=16000)
    for d in ds:
        pass