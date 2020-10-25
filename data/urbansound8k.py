import torch
import os
import logging
import glob
import soundfile as sf
import numpy as np
import resampy
import sys
import torchaudio
from data.data_transformer import Compose, FakePitchShift
from data import register_dataset
import math
from tqdm import tqdm
torchaudio.set_audio_backend("soundfile")  # switch backend

logger = logging.getLogger(__name__)


@register_dataset
class UrbanSound8K(torch.utils.data.Dataset):
    """
    sr is different file to file, most is 44100, 48000, 96000.
    max 4 seconds audio, some less.
    """
    def __init__(self, fold, split, target_sr=44100, transform=None):
        super(UrbanSound8K, self).__init__()
        if target_sr == 44100:
            root = '/data/songhongwei/UrbanSound8K/audio44100/'
        elif target_sr == 22050:
            root = '/data/songhongwei/UrbanSound8K/audio22050/'
        else:
            root = '/data/songhongwei/UrbanSound8K/audio/'
        logger.info("Loading data from {}".format(root))
        if not os.path.exists(root):
            raise Exception("{} does not exists.".format(root))

        feat_files = glob.glob(os.path.join(root, "**/*.wav"), recursive=True)
        feat_set = set(feat_files)
        val_files = glob.glob(os.path.join(root, "fold{}/*.wav".format(str(fold))))
        val_set = set(val_files)
        train_set = feat_set - val_set

        self.all_files = feat_files

        self.target_sr = target_sr

        if split == "train":
            self.files = list(train_set)
        elif split == "valid":
            self.files = val_files
        elif split == "full":
            self.files = feat_files
        else:
            raise ValueError("split not supported.")

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

        with torch.no_grad():
            # (channels, frames)
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
            # print(file, sr)
            audio = self._process_audio(audio, sr, self.target_sr)
            wav_name = os.path.basename(file)
            label = int(wav_name.split('-')[1])

            sample = (audio.requires_grad_(False), torch.as_tensor(label, dtype=torch.int64).requires_grad_(False))

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

        audio = torch.as_tensor(audio, dtype=torch.float32)
        # padding to 4 seconds audio on both side
        target_len = target_sr * 4
        pad_len = target_len - len(audio)
        if pad_len > 0:
            _pad = math.ceil(pad_len / 2)
            if pad_len % 2 == 0:
                audio = torch.nn.functional.pad(audio, [_pad, _pad])
            else:
                audio = torch.nn.functional.pad(audio, [_pad, _pad - 1])

            assert audio.size(0) == target_len

        if audio.size(0) > target_len:
            audio = audio[:target_len]
        return audio.requires_grad_(False)

    def print(self):
        for file in self.all_files:
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
            print(file, sr, audio.size(1) / sr)

    def get_flens(self):
        self.flens = []
        for file in self.all_files:
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
            self.flens.append(audio.size(1) / sr)
        return self.flens


@register_dataset
class UrbanSound8KRepeat(torch.utils.data.Dataset):
    """
    Padding <4s audio to 4s by repeating the recording.
    sr is different file to file, most is 44100, 48000, 96000.
    max 4 seconds audio, some less.
    """
    def __init__(self, fold, split, target_sr=44100, transform=None):
        super(UrbanSound8KRepeat, self).__init__()
        # load pre-saved resampled audio instead of resampling on the fly,
        # which is computational expensive.
        if target_sr == 44100:
            root = '/data/songhongwei/UrbanSound8K/audio44100/'
        elif target_sr == 22050:
            root = '/data/songhongwei/UrbanSound8K/audio22050/'
        else:
            root = '/data/songhongwei/UrbanSound8K/audio/'
        logger.info("Loading data from {}".format(root))
        if not os.path.exists(root):
            raise Exception("{} does not exists.".format(root))

        feat_files = glob.glob(os.path.join(root, "**/*.wav"), recursive=True)
        feat_set = set(feat_files)
        val_files = glob.glob(os.path.join(root, "fold{}/*.wav".format(str(fold))))
        val_set = set(val_files)
        train_set = feat_set - val_set

        self.all_files = feat_files

        self.target_sr = target_sr

        if split == "train":
            self.files = list(train_set)
        elif split == "valid":
            self.files = val_files
        elif split == "full":
            self.files = feat_files
        else:
            raise ValueError("split not supported.")

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

        with torch.no_grad():
            # (channels, frames)
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)

            # print(file, sr)
            audio = self._process_audio(audio, sr, self.target_sr)
            wav_name = os.path.basename(file)
            label = int(wav_name.split('-')[1])

            sample = (audio.requires_grad_(False), torch.as_tensor(label, dtype=torch.int64).requires_grad_(False))

        if self.transform and not isinstance(self.transform, FakePitchShift):
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.files)

    def _process_audio(self, audio, sr, target_sr):
        if audio.size(0) == 2:
            # Downmix if multichannel
            audio = torch.mean(audio, dim=0, keepdim=True)

        audio = audio[0]

        if sr != target_sr:
            audio = resampy.resample(audio.numpy(), sr_orig=sr, sr_new=target_sr, filter='kaiser_best')

        audio = torch.as_tensor(audio, dtype=torch.float32)

        # repeat no more than 4 times.
        pad_zero_len = target_sr - len(audio)
        if pad_zero_len > 0:
            audio = self.pad_zero(audio, pad_zero_len)

        # padding to 4 seconds audio on both side
        target_len = target_sr * 4
        pad_len = target_len - len(audio)
        if pad_len > 0:
            n_repeat = math.ceil(target_len / len(audio))
            audio = audio.repeat(n_repeat)[:target_len]

        if audio.size(0) > target_len:
            audio = audio[:target_len]
        return audio.requires_grad_(False)

    def pad_zero(self, audio, pad_zero_len):
        if pad_zero_len > 0:
            _pad = math.ceil(pad_zero_len / 2)
            if pad_zero_len % 2 == 0:
                audio = torch.nn.functional.pad(audio, [_pad, _pad])
            else:
                audio = torch.nn.functional.pad(audio, [_pad, _pad - 1])
        return audio

    def print(self):
        """
        print file name, sr, and length in seconds.
        """
        for file in self.all_files:
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
            print(file, sr, audio.size(1) / sr)

    def get_flens(self):
        """
        file length in seconds.
        """
        self.flens = []
        for file in self.all_files:
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
            self.flens.append(audio.size(1) / sr)
        return self.flens

    def raw_by_id(self, idx):
        """
        load raw wave by file idx.
        """
        audio, sr = torchaudio.load(filepath=self.files[idx], normalization=lambda x: torch.abs(x).max(),
                                    channels_first=True)
        wavname = os.path.basename(self.files[idx])
        return audio[0], wavname.split('-')[1]


@register_dataset
class UrbanSound8KCached(torch.utils.data.Dataset):
    """
    cache audio wav into memory, so that no need for resampling at each batch.
    """
    def __init__(self, fold, split, target_sr=44100, transform=None):
        super(UrbanSound8KCached, self).__init__()
        root = '/data/songhongwei/UrbanSound8K/audio/'
        if not os.path.exists(root):
            raise Exception("{} does not exists.".format(root))

        feat_files = glob.glob(os.path.join(root, "**/*.wav"), recursive=True)
        feat_set = set(feat_files)
        val_files = glob.glob(os.path.join(root, "fold{}/*.wav".format(str(fold))))
        val_set = set(val_files)
        train_set = feat_set - val_set

        self.all_files = feat_files

        self.target_sr = target_sr

        if split == "train":
            self.files = list(train_set)
        elif split == "valid":
            self.files = val_files
        elif split == "full":
            self.files = feat_files
        else:
            raise ValueError("split not supported.")

        self.transform = transform
        logger.info("Loading fold {}, split {}, {} files".format(str(fold), split, len(self.files)))
        self.samples = self._cache_audio()

    def _cache_audio(self):
        samples = []
        for file in tqdm(self.files):
            # (channels, frames)
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
            # print(file, sr)
            audio = self._process_audio(audio, sr, self.target_sr)
            wav_name = os.path.basename(file)
            label = int(wav_name.split('-')[1])
            samples.append((audio.requires_grad_(False),
                            torch.as_tensor(label, dtype=torch.int64).requires_grad_(False)))
        return samples

    def __getitem__(self, idx):
        """
        :param idx:
        :return: audio, label
        """
        return self.samples[idx]

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

        audio = torch.as_tensor(audio, dtype=torch.float32)
        # padding to 4 seconds audio on both side
        target_len = target_sr * 4
        pad_len = target_len - len(audio)
        if pad_len > 0:
            _pad = math.ceil(pad_len / 2)
            if pad_len % 2 == 0:
                audio = torch.nn.functional.pad(audio, [_pad, _pad])
            else:
                audio = torch.nn.functional.pad(audio, [_pad, _pad - 1])

            assert audio.size(0) == target_len

        if audio.size(0) > target_len:
            audio = audio[:target_len]
        return audio.requires_grad_(False)

    def print(self):
        for file in self.all_files:
            audio, sr = torchaudio.load(filepath=file, normalization=lambda x: torch.abs(x).max(), channels_first=True)
            print(file, sr, audio.size(1) / sr)


if __name__ == '__main__':
    dataset = UrbanSound8KRepeat(fold=2, split='train', target_sr=22050)
    # dataset.output_sr_length()
    # for data in dataset:
    # dataset.print()
    for d in dataset:
        pass