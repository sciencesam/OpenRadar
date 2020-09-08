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

from pathlib import Path
import matplotlib.pyplot  as plt
import xml.etree.ElementTree as ET
import numpy as np
import struct
import glob


# def parse_raw_adc(source_fp, dest_fp):
    # """Reads a binary data file containing raw adc data from a DCA1000, cleans it and saves it for manual processing.

    # Note:
    #     "Raw adc data" in this context refers to the fact that the DCA1000 initially sends packets of data containing
    #     meta data and is merged with actual pure adc data. Part of the purpose of this function is to remove this
    #     meta data.

    # Args:
    #     source_fp (str): Path to raw binary adc data.
    #     dest_fp (str): Path to output cleaned binary adc data.

    # Returns:
    #     None

    # """

xml_file = Path("I:\\IEEE_Radar_Contest\\Data_Collect\\20200714_SarAttempt2\\SARsetup.xml")
data_files = glob.glob("I:\\IEEE_Radar_Contest\\Data_Collect\\20200714_SarAttempt2\SAR*.bin");

# Get Configuration Values
config_root = ET.parse(xml_file)

# TODO : read this value from config
num_channels = 4
num_streams = num_channels * 2
adc_samples = int(config_root.find(".//*[@name='numAdcSamples']").get('value'))
pulses_in_frame = int(config_root.find(".//*[@name='loopCount']").get('value'))
num_frames = int(config_root.find(".//*[@name='frameCount']").get('value'))

raw_sar_data = np.zeros((len(data_files) ,num_channels, pulses_in_frame, adc_samples), dtype = "complex_")

for data_iter, data_file in enumerate(data_files):

    # Read in binary
    buff = np.fromfile(data_file, dtype=np.uint16)

    # Change to iq data streams
    iq_streams = buff.reshape(pulses_in_frame * adc_samples, num_streams)

    # Change to channel values
    channel_stream = np.zeros((num_channels, pulses_in_frame * adc_samples), dtype = "complex_")
    for channel_iter in range(0, num_channels):
        channel_stream[channel_iter, :] = iq_streams[:, 2*channel_iter] +  1j * iq_streams[:, 2*channel_iter + 1]

    # Change to data cube
    data_cube = channel_stream.reshape((num_channels, pulses_in_frame, adc_samples))

    raw_sar_data[data_iter ,: ,:, :] = data_cube

# fft across range and just choose a channel and pick the first pulse for now

new_data_cube = raw_sar_data[:, 0, pulses_in_frame - 1 , :]

range_compressed_cube = np.fft.fft(new_data_cube, adc_samples)