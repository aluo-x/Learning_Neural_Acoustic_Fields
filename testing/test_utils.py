import numpy as np
import pyroomacoustics
import librosa
from scipy.io import wavfile

def spectral(in1, in2):
    return np.mean(np.abs(in1-in2))

def to_wave(input_spec, mean_val=None, std_val=None, gl=False, orig_phase=None):
    if not mean_val is None:
        renorm_input = input_spec * std_val
        renorm_input = renorm_input + mean_val
    else:
        renorm_input = input_spec + 0.0
    renorm_input = np.concatenate((renorm_input, renorm_input[-1:]*0.0), axis=0)
    if orig_phase is None:
        if gl == False:
            # Random phase reconstruction per image2reverb
            # do not use griffinlim
            np.random.seed(1234)
            rp = np.random.uniform(-np.pi, np.pi, renorm_input.shape)
            f = renorm_input * (np.cos(rp) + (1.j * np.sin(rp)))
            out_wave = librosa.istft(f)
        else:
            out_wave = librosa.griffinlim(renorm_input, win_length=512, hop_length=128, n_iter=100, momentum=0.5, random_state=64)
    else:
        f = renorm_input * (np.cos(orig_phase) + (1.j * np.sin(orig_phase)))
        out_wave = librosa.istft(f)
    return np.clip(out_wave, -1, 1)

def get_wave(gen_spec):
    sig_0 = to_wave(gen_spec[0], gl=False)
    sig_1 = to_wave(gen_spec[1], gl=False)
    return sig_0, sig_1

def compute_t60(true_in, gen_in):
    final_container = []
    try:
        left_true = pyroomacoustics.experimental.measure_rt60(true_in[0], fs=22050, decay_db=30)
        right_true = pyroomacoustics.experimental.measure_rt60(true_in[1], fs=22050, decay_db=30)
        left_gen = pyroomacoustics.experimental.measure_rt60(gen_in[0], fs=22050, decay_db=30)
        right_gen = pyroomacoustics.experimental.measure_rt60(gen_in[1], fs=22050, decay_db=30)
    except:
        left_true = -1
        right_true = -1
        left_gen = -1
        right_gen = -1
    to_return = [left_true, right_true, left_gen, right_gen]
    return to_return

def load_audio(path_name, resample_rate=22050):
    # returns in shape (ch, num_sample), as float32 (on Linux at least)
    # by default torchaudio is wav_arr, sample_rate
    # by default wavfile is sample_rate, wav_arr
    loaded = wavfile.read(path_name)
    dtp = loaded[1].dtype
    wave_data_loaded = np.clip(loaded[1], -1.0, 1.0).T
    sr_loaded = loaded[0]
    resampled_wave = librosa.resample(wave_data_loaded, orig_sr=sr_loaded, target_sr=resample_rate)
    return np.clip(resampled_wave, -1.0, 1.0)


class get_spec():
    def __init__(self, use_torch=False, power_mod=2, fft_size=512):
        self.n_fft = fft_size
        self.hop = self.n_fft // 4
        if use_torch:
            import torch
            from torchaudio.transforms import Spectrogram
            self.use_torch = True
            #             self.spec_transform = Spectrogram(power=None)
            self.spec_transform = Spectrogram(power=None, n_fft=self.n_fft, hop_length=self.hop)
        else:
            self.power = power_mod
            self.use_torch = False
            self.spec_transform = None

    def transform(self, wav_data_prepad):
        wav_data = librosa.util.fix_length(wav_data_prepad, wav_data_prepad.shape[-1] + self.n_fft // 2)
        if wav_data.shape[1] < 4410:
            wav_data = librosa.util.fix_length(wav_data, 4410)
        if self.use_torch:
            transformed_data = self.spec_transform(torch.from_numpy(wav_data)).numpy()
        else:
            transformed_data = np.array([librosa.stft(wav_data[0], n_fft=self.n_fft, hop_length=self.hop),
                                         librosa.stft(wav_data[1], n_fft=self.n_fft, hop_length=self.hop)])[:, :-1]

        real_component = np.abs(transformed_data)
        return np.log(real_component + 1e-3)
