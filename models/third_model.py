from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader


class AMISpeakerDataset(Dataset):
    def __init__(self, audio_dir, rttm_dir, speaker_to_id, window_size=3.0, window_hop=1.5,
                 sample_rate=16000, min_segment_duration=1.0):
        self.audio_dir = Path(audio_dir)
        self.rttm_dir = Path(rttm_dir)
        self.speaker_to_id = speaker_to_id
        self.window_size = window_size
        self.window_hop = window_hop
        self.sample_rate = sample_rate
        self.window_samples = int(window_size * sample_rate)
        self.min_samples = int(min_segment_duration * sample_rate)

        self.chunks = self._prepare_data()

    def _parse_rttm(self, rttm_path):
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    duration = float(parts[4])
                    spk_id = parts[7]

                    if spk_id in self.speaker_to_id:
                        segments.append((start, duration, spk_id))
        return segments

    def _prepare_data(self):
        chunks = []
        audio_files = list(self.audio_dir.glob('*.wav'))

        for audio_path in audio_files:
            file_id = audio_path.stem.split('.')[0]
            rttm_path = self.rttm_dir / f"{file_id}.rttm"

            if not rttm_path.exists():
                continue

            speaker_segments = self._parse_rttm(rttm_path)

            for seg_start, duration, spk_id in speaker_segments:
                seg_end = seg_start + duration
                current_time = seg_start

                if duration >= self.window_size:
                    while current_time + self.window_size <= seg_end:
                        chunks.append({
                            'audio_path': audio_path,
                            'start_time': current_time,
                            'duration': self.window_size,
                            'speaker_label': self.speaker_to_id[spk_id]
                        })
                        current_time += self.window_hop

                elif duration >= (self.min_samples / self.sample_rate):
                    chunks.append({
                        'audio_path': audio_path,
                        'start_time': seg_start,
                        'duration': duration,
                        'speaker_label': self.speaker_to_id[spk_id]
                    })

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        frame_offset = int(chunk['start_time'] * self.sample_rate)
        num_frames = int(chunk['duration'] * self.sample_rate)

        waveform, sr = torchaudio.load(
            chunk['audio_path'],
            frame_offset=frame_offset,
            num_frames=num_frames
        )

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if waveform.shape[1] < self.window_samples:
            pad_len = self.window_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :self.window_samples]

        label = torch.tensor(chunk['speaker_label'], dtype=torch.long)

        return waveform, label


class AMISpeakerDataModule(pl.LightningDataModule):
    def __init__(self, audio_dir, train_rttm_dir, val_rttm_dir, window_size=3.0, batch_size=64, num_workers=4):
        super().__init__()
        self.audio_dir = audio_dir
        self.train_rttm_dir = train_rttm_dir
        self.val_rttm_dir = val_rttm_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.speaker_to_id = self._build_speaker_vocab()
        self.num_classes = len(self.speaker_to_id)

    def _build_speaker_vocab(self):
        unique_speakers = set()
        for rttm_dir in [self.train_rttm_dir, self.val_rttm_dir]:
            if rttm_dir and Path(rttm_dir).exists():
                for rttm_path in Path(rttm_dir).glob('*.rttm'):
                    with open(rttm_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 8 and parts[0] == "SPEAKER":
                                unique_speakers.add(parts[7])

        return {spk: i for i, spk in enumerate(sorted(list(unique_speakers)))}

    def setup(self, stage=None):
        self.train_dataset = AMISpeakerDataset(
            self.audio_dir, self.train_rttm_dir,
            speaker_to_id=self.speaker_to_id, window_size=self.window_size
        )
        self.val_dataset = AMISpeakerDataset(
            self.audio_dir, self.val_rttm_dir,
            speaker_to_id=self.speaker_to_id, window_size=self.window_size
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers)


class SpeakerEmbeddingExtractor(nn.Module):
    def __init__(self, embedding_dim=256, sample_rate=16000):
        super().__init__()

        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=64
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        self.resnet = models.resnet18(weights=None)

        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def forward(self, x):
        x = self.mel_spectrogram(x)
        x = self.amplitude_to_db(x)

        embedding = self.resnet(x)

        return F.normalize(embedding, p=2, dim=1)


class SpeakerLightningModule(pl.LightningModule):
    def __init__(self, num_speakers, embedding_dim=256, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.embedding_extractor = SpeakerEmbeddingExtractor(embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_speakers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.embedding_extractor(x)

    def training_step(self, batch, batch_idx):
        waveforms, labels = batch
        embeddings = self.embedding_extractor(waveforms)
        outputs = self.classifier(embeddings)

        loss = self.loss_fn(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        waveforms, labels = batch

        embeddings = self.embedding_extractor(waveforms)

        sim_matrix = torch.mm(embeddings, embeddings.t())

        target_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        diag_mask = 1.0 - torch.eye(sim_matrix.size(0), device=self.device)
        sim_vector = sim_matrix[diag_mask == 1]
        target_vector = target_matrix[diag_mask == 1]

        pos_sim = sim_vector[target_vector == 1].mean() if (target_vector == 1).any() else torch.tensor(0.0)
        neg_sim = sim_vector[target_vector == 0].mean() if (target_vector == 0).any() else torch.tensor(0.0)

        val_proxy_loss = 1.0 - (pos_sim - neg_sim)

        self.log('val_proxy_loss', val_proxy_loss, on_epoch=True, prog_bar=True)
        self.log('val_pos_speaker_sim', pos_sim, on_epoch=True, prog_bar=True)
        self.log('val_neg_speaker_sim', neg_sim, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        return optimizer


if __name__ == '__main__':
    AUDIO_DIR = '../pyannote/amicorpus'
    TRAIN_RTTM_DIR = '../only_words/rttms/train'
    VAL_RTTM_DIR = '../only_words/rttms/dev'

    data_module = AMISpeakerDataModule(audio_dir=AUDIO_DIR, train_rttm_dir=TRAIN_RTTM_DIR, val_rttm_dir=VAL_RTTM_DIR,
                                       batch_size=32)

    model = SpeakerLightningModule(num_speakers=data_module.num_classes, embedding_dim=256, lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="cuda",
        devices=1,
    )

    trainer.fit(model, datamodule=data_module)
