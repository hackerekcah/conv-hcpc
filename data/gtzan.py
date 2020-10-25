import torch
import os
import logging
import glob
import soundfile as sf
import numpy as np
import resampy
import torchaudio
from data import register_dataset
from data.data_transformer import Compose, FakePitchShift
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
torchaudio.set_audio_backend("soundfile")  # switch backend

logger = logging.getLogger(__name__)


@register_dataset
class GTZAN(torch.utils.data.Dataset):
    """
    sr=22050, clip to 30 seconds, mono
    """
    def __init__(self, fold, split, target_sr=22050, transform=None):
        """
        :param fold: 1 to 5
        :param split:
        :param target_sr:
        :param transform:
        """
        super(GTZAN, self).__init__()
        root = '/data/songhongwei/genres/'
        if not os.path.exists(root):
            raise Exception("{} does not exists.".format(root))

        self.target_sr = target_sr

        feat_files = glob.glob(os.path.join(root, "**/*.wav"), recursive=True)

        labels_str = [os.path.basename(p).split('.')[0] for p in feat_files]
        le = LabelEncoder()
        labels = le.fit_transform(labels_str)

        # skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=None)
        # set random_state to a fixed value to get fixed fold splits
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        # list of tuples, each tuple contains (train_idx, test_idx) of the corresponding fold
        splits_idx = list(skf.split(np.zeros(len(labels)), labels))

        feat_files = np.array(feat_files)

        if split == 'train':
            self.files = feat_files[splits_idx[fold-1][0]]
            self.labels = labels[splits_idx[fold-1][0]]
        else:
            self.files = feat_files[splits_idx[fold-1][1]]
            self.labels = labels[splits_idx[fold-1][1]]

        self.transform = transform

        logger.info("Loading fold {}, split {}, {} files".format(str(fold), split, len(self.files)))

    def __getitem__(self, idx):
        """
        :param idx:
        :return: audio, label
        """
        file = self.files[idx]
        file = self.files[idx]
        if isinstance(self.transform, FakePitchShift):
            file = self.transform(file)
        if isinstance(self.transform, Compose):
            if isinstance(self.transform.transforms[0], FakePitchShift):
                file = self.transform.transforms[0](file)
        # (channels, frames)
        audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
        audio = self._process_audio(audio, sr, self.target_sr)
        label = self.labels[idx]
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

        if len(audio) > target_sr * 30:
            audio = audio[: target_sr*30]
        elif len(audio) < target_sr * 30:
            audio = torch.as_tensor(audio, dtype=torch.float32)
            # padding to 30 seconds audio on both side
            target_len = target_sr * 30
            pad_len = target_len - len(audio)
            if pad_len > 0:
                _pad = math.ceil(pad_len / 2)
                if pad_len % 2 == 0:
                    audio = torch.nn.functional.pad(audio, [_pad, _pad])
                else:
                    audio = torch.nn.functional.pad(audio, [_pad, _pad - 1])

                assert audio.size(0) == target_len

        return audio.requires_grad_(False)


if __name__ == '__main__':
    data = GTZAN(fold=1, split='train', target_sr=22050)
    data2 = GTZAN(fold=1, split='test', target_sr=22050)
