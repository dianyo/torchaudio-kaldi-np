import kaldi_np 
import torchaudio.compliance.kaldi as kaldi
import torch
import numpy as np

test_np = kaldi_np._feature_window_function('hanning', 400, 0.42)
test_torch = kaldi._feature_window_function('hanning', 400, 0.42, torch.device(type='cpu'), torch.float32)
print("_feature_window_function", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

test_wav = np.random.rand(16000)
strided_input_np = kaldi_np._get_strided(test_wav, 400, 160, True)
strided_input_torch = kaldi._get_strided(torch.from_numpy(test_wav), 400, 160, True)
print("_get_strided", np.allclose(strided_input_np, strided_input_torch.numpy()))

test_np = kaldi_np._get_log_energy(strided_input_np, 1.1921e-07, 1.0)
test_torch = kaldi._get_log_energy(strided_input_torch, torch.Tensor([1.1921e-07]), 1.0)
print("_get_log_energy", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

test_np = kaldi_np._get_waveform_and_window_properties(test_wav.reshape((1, -1)), -1, 16000, 10, 25.0, True, 0.97)
test_torch = kaldi._get_waveform_and_window_properties(torch.from_numpy(test_wav.reshape((1, -1))), -1, 16000, 10, 25.0, True, 0.97)
print("_get_waveform_and_window_properties", np.allclose(test_np[0], test_torch[0].numpy(), rtol=1e-05, atol=1e-05))

test_np = kaldi_np._get_window(test_wav, 512, 400, 160, 'hanning', 0.42, True, True, 1.0, 0.0, True, 0.97)
test_torch = kaldi._get_window(torch.from_numpy(test_wav), 512, 400, 160, 'hanning', 0.42, True, True, 1.0, 0.0, True, 0.97)
print("_get_window1", np.allclose(test_np[0], test_torch[0].numpy(), rtol=1e-05, atol=1e-05))
print("_get_window2", np.allclose(test_np[1], test_torch[1].numpy(), rtol=1e-05, atol=1e-05))

test_np_input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
test_np = kaldi_np._subtract_column_mean(test_np_input, True)
test_torch = kaldi._subtract_column_mean(torch.from_numpy(test_np_input), True)
print("_subtract_column_mean", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

test_waveform = np.random.randn(1, 16000)
test_np = kaldi_np.spectrogram(test_waveform)
test_torch = kaldi.spectrogram(torch.from_numpy(test_waveform))
print("spectrogram", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

mel_freq = np.array([100.0, 200.0, 300.0])
test_np = kaldi_np.inverse_mel_scale(mel_freq)
test_torch = kaldi.inverse_mel_scale(torch.from_numpy(mel_freq))
print("inverse_mel_scale", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

test_freq = np.array([1000.0, 2000.0, 3000.0])
test_np = kaldi_np.mel_scale(test_freq)
test_torch = kaldi.mel_scale(torch.from_numpy(test_freq))
print("mel_scale", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

vtln_low_cutoff = 100.0
vtln_high_cutoff = 1000.0
low_freq = 50.0
high_freq = 2000.0
vtln_warp_factor = 1.5
freq = np.array([75.0, 150.0, 500.0, 1200.0, 1800.0])

test_np = kaldi_np.vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, freq)
test_torch = kaldi.vtln_warp_freq(vtln_low_cutoff, vtln_high_cutoff, low_freq, high_freq, vtln_warp_factor, torch.from_numpy(freq))
print("vtln_warp_freq", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

num_bins = 40
window_length_padded = 512
sample_freq = 16000.0
low_freq = 0.0
high_freq = 8000.0
vtln_low = 100.0
vtln_high = 6000.0
vtln_warp_factor = 1

test_np = kaldi_np.get_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq, vtln_low, vtln_high, vtln_warp_factor)
test_torch = kaldi.get_mel_banks(num_bins, window_length_padded, sample_freq, low_freq, high_freq, vtln_low, vtln_high, vtln_warp_factor)

print("vtln_warp_freq1", np.allclose(test_np[0], test_torch[0].numpy(), rtol=1e-05, atol=1e-05))
print("vtln_warp_freq2", np.allclose(test_np[1], test_torch[1].numpy(), rtol=1e-05, atol=1e-05))

waveform = np.random.rand(2, 16000)
test_np = kaldi_np.fbank(waveform)
test_torch = kaldi.fbank(torch.from_numpy(waveform))
print("fbank", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))

num_ceps = 13
num_mel_bins = 23

test_np = kaldi_np._get_dct_matrix(num_ceps, num_mel_bins)
test_torch = kaldi._get_dct_matrix(num_ceps, num_mel_bins)
print("_get_dct_matrix", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))
num_ceps = 13
cepstral_lifter = 22

test_np = kaldi_np._get_lifter_coeffs(num_ceps, cepstral_lifter)
test_torch = kaldi._get_lifter_coeffs(num_ceps, cepstral_lifter)
print("_get_lifter_coeffs", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-05))
waveform = np.random.rand(2, 16000)  # Example waveform of shape (2, 16000)

# Call the converted function
test_np = kaldi_np.mfcc(waveform)
test_torch = kaldi.mfcc(torch.from_numpy(waveform))
print("mfcc", np.allclose(test_np, test_torch.numpy(), rtol=1e-05, atol=1e-04))
