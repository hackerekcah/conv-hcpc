import torch
import math
import pyrubberband as pyrb
import librosa
import random


class RandomCropWav:
    def __init__(self, target_sr, crop_seconds):
        self.target_sr = target_sr
        self.len = math.floor(self.target_sr * crop_seconds)

    def __call__(self, sample):
        x = sample[0]
        start = torch.randint(low=0, high=x.size(0)-self.len, size=())
        x = x[start: start + self.len]
        return x, sample[1]


class PitchShift:
    """
    Pitch shifting wave on the fly, computational expensive.
    """
    def __init__(self, target_sr, pitch_shift_steps):
        """
        :param target_sr:
        :param pitch_shift_steps: a list of semitones (float)
        """
        self.target_sr = target_sr
        # a list of steps
        self.pitch_shift_steps = pitch_shift_steps

    def __call__(self, sample):
        # if empty list
        if len(self.pitch_shift_steps) == 0:
            return sample
        x = sample[0]
        i = random.randint(0, len(self.pitch_shift_steps) - 1)
        semitones = self.pitch_shift_steps[i]
        if semitones == 0:
            return sample
        else:
            y = librosa.effects.pitch_shift(x.numpy(), sr=self.target_sr, n_steps=float(semitones))
            # y = pyrb.pitch_shift(x.numpy(), sr=self.target_sr, n_steps=float(semitones))
            return torch.as_tensor(y, dtype=torch.float32), sample[1]


class FakePitchShift:
    """
    load pre-shifted audio, bcz pitch shift on the fly is computational expensive.
    """
    def __init__(self, target_sr, pitch_shift_steps):
        """
        :param target_sr:
        :param pitch_shift_steps: a list of semitones (float)
        """
        self.target_sr = target_sr
        # a list of steps
        self.pitch_shift_steps = pitch_shift_steps

    def __call__(self, filepath):
        # if empty list
        if len(self.pitch_shift_steps) == 0:
            return filepath
        i = random.randint(0, len(self.pitch_shift_steps) - 1)
        semitones = self.pitch_shift_steps[i]
        if semitones == 0:
            return filepath
        else:
            path_splits = filepath.split('/')
            if "ESC" in filepath:
                path_splits[-2] = 'audio-ps' + "{:.1f}".format(semitones)
            elif "genres" in filepath:
                path_splits[-3] = 'genres-ps' + "{:.1f}".format(semitones)
            elif "UrbanSound8K" in filepath:
                path_splits[-3] = "{}-ps{:.1f}".format(path_splits[-3], semitones)
            ps_path = '/'.join(path_splits)
            return ps_path


class Compose:
    def __init__(self, transforms):
        """
        :param transforms: list of transforms, item in list can be None
        """
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            if t is not None and not isinstance(t, FakePitchShift):
                sample = t(sample)
        return sample


class TimeStretch:
    """
    Time-stretching, then cropping or zero-padding to keep wav length not changed.
    """
    def __init__(self, target_sr, stretch_args):
        """
        :param target_sr:
        :param stretch_range: [prob, low, high]
        [0, 1, 1]: no time stretch
        """
        # self.prob percent times use stretch.
        self.prob = stretch_args[0]
        self.low = stretch_args[1]
        self.high = stretch_args[2]
        self.target_sr = target_sr

    def __call__(self, sample):
        if self.low == 1 and self.high == 1 or self.prob == 0:
            return sample

        if random.uniform(0, 1) > self.prob:
            return sample

        rate = random.uniform(self.low, self.high)
        x = sample[0]
        target_len = x.size(0)
        y = librosa.effects.time_stretch(x.numpy(), rate=rate)
        # y = pyrb.time_stretch(x.numpy(), sr=self.target_sr, rate=rate)
        audio = self._keep_len(y, target_len=target_len)
        return audio, sample[1]

    def _keep_len(self, audio, target_len):
        audio = torch.as_tensor(audio, dtype=torch.float32)
        if len(audio) > target_len:
            diff = len(audio) - target_len
            start = math.floor(diff / 2)
            audio = audio[start: target_len + start]
        elif len(audio) < target_len:
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
    # r = RandomCropWav(target_sr=10, crop_seconds=3)
    # r((torch.randn(1000,), 1))
    #
    # ps = PitchShift(target_sr=8000, pitch_shift_steps=[-2, -1, 1, 2])
    # for _ in range(5):
    #     s = ps((torch.randn(16000,), 1))
    #     print(s[0])
    #
    # t = Compose([None, ps])
    # t((torch.randn(16000,), 1))
    #
    # ts = TimeStretch(target_sr=8000, stretch_args=[0.8, 0.8, 1.2])
    # for _ in range(10):
    #     sample = ts((torch.randn(16000,), 1))
    #     print(sample[0])

    fps = FakePitchShift(target_sr=8000, pitch_shift_steps=[-2, -1, 0, 1, 2])
    print(fps.__name__)
    for _ in range(5):
        fname = fps('/data/songhongwei/ESC-50/audio/3-144106-A-32.wav')
        print(fname)
