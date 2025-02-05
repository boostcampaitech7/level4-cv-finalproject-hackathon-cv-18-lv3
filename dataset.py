# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import soundfile as sf
import numpy as np
from transformers import WhisperFeatureExtractor
import librosa
import torchaudio.compliance.kaldi as ta_kaldi

class SALMONNDataset(Dataset):
    def __init__(self, prefix, ann_path, whisper_path):
        super().__init__()

        self.prefix = prefix

        if ann_path is not None and ann_path != "":
            self.annotation = json.load(open(ann_path, "r"))["annotation"]
        else:
            self.annotation = []

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

        self.beats_num_mel_bins = 128
        self.beats_fbank_mean = 15.41663
        self.beats_fbank_std = 6.55582

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        beats_specs = [s["beats_spectrogram"] for s in samples]
        cat_beats_spec = torch.stack(beats_specs, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "spectrogram": cat_spectrogram,
            "beats_spectrogram": cat_beats_spec,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,
            "task": task,
            "Q": Q,
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_path = self.prefix + ann["path"] if ann["path"].startswith("/") else self.prefix + "/" + ann["path"]
        try:
            audio, sr = sf.read(audio_path)
        except:
            print(f"Failed to load {audio_path}. Load 0-th sample for now")
            audio, sr = sf.read(self.prefix + '/' + self.annotation[0]["path"])
        
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]

        #if "expand_wav" in ann:
            #for p in ann["expand_wav"]:
                #expand_audio, _ = sf.read(self.prefix + '/' + p)
                #if len(expand_audio.shape) == 2:
                    #expand_audio = expand_audio[:, 0]
                #sil = np.zeros(int(sr/10), dtype=float)
                #audio = np.concatenate((audio, sil, expand_audio), axis=0)
                
        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)

        if sr != self.wav_processor.sampling_rate: # TODO. use more efficient implementation            
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.wav_processor.sampling_rate)
            sr = self.wav_processor.sampling_rate

        audio = audio[: sr * 30] # truncate audio to at most 30s

        # spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        # whisper용 80-bin mel spectrogram 추출
        whisper_spec = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        
        # BEATs용 128-bin mel spectrogram 추출
        waveform = torch.from_numpy(audio).float()
        waveform_batch = waveform.unsqueeze(0)
        beats_spec = self.preprocess(waveform_batch, fbank_mean=self.beats_fbank_mean, fbank_std=self.beats_fbank_std)
        beats_spec = (beats_spec - self.beats_fbank_mean) / (2 * self.beats_fbank_std)
        
        text = ann["text"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        return {
            "spectrogram": whisper_spec,
            "beats_spectrogram": beats_spec,
            "raw_wav": audio,
            "text": text,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }