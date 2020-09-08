# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import matplotlib as plt
from . import utils

#
# The following code needs to be put set into a configuration folder
#

nfft_time = 1024             # Number of FFT points for Range-FFT
target_range = 280e-3        # Range of target (range of corresponding image slice) in meters
delta_x = 200/406 * 1e-3     # Sampling distance at x (horizontal) axis in meters
delta_y = 2 * 1e-3           # Sampling distance at y (vertical) axis in meters
nfft_space = 1024            # Number of FFT points for Spatial-FFT
sample_freq = 9121e3         # Sampling rate (sps)
delta_time = 1/sample_freq   # Sampling period
freq_slope = 63.343e12       # Slope const (Hz/sec)
speed_of_light = 3e8         # Speed of Light in meters per second
freq_start = 77e9            # Starting frequency

#
# Pull out SAR data for slice in range
#

raw_data_fft = np.fft.fft(raw_data, nfft_space)
range_bin = 2 * freq_slope * delta_time * target_range * nfft_time / speed_of_light
sar_data = raw_data_fft[range_bin, : ,:]

#
# Create Matched Filter
#

x_axis = delta_x * np.arange(start=((-nfft_space - 1)/2), stop=((nfft_space - 1)/2), step=1)
y_axis = delta_y * np.arange(start=((-nfft_space - 1)/2), stop=((nfft_space - 1)/2), step=1)

omega = 2 * np.pi * freq_start / speed_of_light # tone of matched filter
matched_filter = exp(-1j * 2 * omega * sqrt(x_axis^2 + y_axis^2 + target_range^2))

#
# Process SAR
#

data_rows = sar_data.shape[0]
data_cols = sar_data.shape[1]

filter_rows = matched_filter.shape[0]
filter_cols = matched_filter.shape[1]

if filter_rows > data_rows:
    pad_length = (filter_rows - data_rows)/ 2
    sar_data = np.pad(sar_data, [ (np.floor(pad_length), 0), (0, 0) ])
    sar_data = np.pad(sar_data, [ (0, np.ceil(pad_length)), (0, 0) ])
else:
    pad_length = (data_rows - filter_rows)/ 2
    matched_filter = np.pad(matched_filter, [ (np.floor(pad_length), 0), (0, 0) ])
    matched_filter = np.pad(matched_filter, [ (0, np.ceil(pad_length)), (0, 0) ])

if filter_cols > data_cols:
    pad_length = (filter_cols - data_cols)/ 2
    sar_data = np.pad(sar_data, [ (0, 0), (np.floor(pad_length), 0) ])
    sar_data = np.pad(sar_data, [ (0, 0), (0, np.ceil(pad_length)) ])
else:
    pad_length = (data_cols - filter_cols)/ 2
    matched_filter = np.pad(matched_filter, [ (0, 0), (np.floor(pad_length), 0) ])
    matched_filter = np.pad(matched_filter, [ (0, 0), (0, np.ceil(pad_length)) ])

sar_data_fft = np.fft.fft2(sar_data)
matched_filter_fft = np.fft.fft2(matched_filter)
sar_image = np.fft.fftshift(np.fft.ifft2(sar_data_fft * matched_filter_fft))

plt.plot_surface(sar_image)