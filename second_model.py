from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy


class AMISegmentationDataset(Dataset):
    def __init__(self, audio_dir, rttm_dir, window_size=2.0, window_hop=1.0,
                 sample_rate=16000, feature_hop_ms=10, max_speakers=4):
        """
        Args:
            audio_dir: Шлях до папки з wav файлами.
            rttm_dir: Шлях до папки з rttm файлами.
            window_size: Розмір аудіо-чанка в секундах.
            window_hop: Крок зсуву вікна при нарізці.
            sample_rate: Частота дискретизації.
            feature_hop_ms: Крок вікна спектрограми в мілісекундах.
            max_speakers: Максимальна кількість спікерів у моделі (для AMI оптимально 4).
        """
        self.audio_dir = Path(audio_dir)
        self.rttm_dir = Path(rttm_dir)
        self.window_size = window_size
        self.window_hop = window_hop
        self.sample_rate = sample_rate
        self.max_speakers = max_speakers

        self.frames_per_window = int(window_size * 1000 / feature_hop_ms)
        self.feature_hop_sec = feature_hop_ms / 1000.0
        self.window_samples = int(window_size * sample_rate)

        self.chunks = self._prepare_data()

    def _parse_rttm(self, rttm_path):
        """Парсить RTTM файл і повертає сегменти із зафіксованими ID спікерів."""
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Формат RTTM: SPEAKER file_id channel start_time duration ... speaker_id
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]  # Отримуємо унікальний ID спікера
                    segments.append((speaker_id, start, start + duration))
        return segments

    def _prepare_data(self):
        """Сканує файли, створює локальний мапінг спікерів та нарізає аудіо."""
        chunks = []
        audio_files = list(self.audio_dir.glob('*.wav'))

        for audio_path in audio_files:
            file_id = audio_path.stem.split('.')[0]
            rttm_path = self.rttm_dir / f"{file_id}.rttm"

            if not rttm_path.exists():
                print(f"Warning: Не знайдено RTTM для {file_id}. Пропускаємо.")
                continue

            # Отримуємо сегменти мовлення (тепер вони містять speaker_id)
            speech_segments = self._parse_rttm(rttm_path)
            if not speech_segments:
                continue

            # Створюємо локальний мапінг спікерів для цього конкретного файлу.
            # Оскільки ми вчимося з PIT, фіксований глобальний індекс не потрібен.
            unique_speakers = sorted(list(set(spk for spk, _, _ in speech_segments)))
            speaker_to_idx = {spk: i for i, spk in enumerate(unique_speakers) if i < self.max_speakers}

            info = torchaudio.info(audio_path)
            total_duration = info.num_frames / info.sample_rate

            current_time = 0.0
            while current_time + self.window_size <= total_duration:
                chunks.append({
                    'audio_path': audio_path,
                    'start_time': current_time,
                    'speech_segments': speech_segments,
                    'speaker_to_idx': speaker_to_idx  # Передаємо мапінг для кожного чанка
                })
                current_time += self.window_hop

        return chunks

    def _generate_labels(self, chunk_start, speech_segments, speaker_to_idx):
        """Генерує матрицю міток форми (Frames, Max_Speakers)."""
        # Створюємо матрицю з нулями
        labels = torch.zeros(self.frames_per_window, self.max_speakers)
        chunk_end = chunk_start + self.window_size

        for spk_id, seg_start, seg_end in speech_segments:
            # Якщо цього спікера немає в мапінгу (перевищено max_speakers) або сегмент поза чанком — пропускаємо
            if spk_id not in speaker_to_idx or seg_end <= chunk_start or seg_start >= chunk_end:
                continue

            spk_idx = speaker_to_idx[spk_id]

            # Обрізаємо межі сегмента до меж поточного чанка
            overlap_start = max(chunk_start, seg_start)
            overlap_end = min(chunk_end, seg_end)

            # Переводимо час в індекси фреймів
            start_frame = int((overlap_start - chunk_start) / self.feature_hop_sec)
            end_frame = int((overlap_end - chunk_start) / self.feature_hop_sec)

            # Захист від виходу за межі масиву через округлення
            start_frame = max(0, min(start_frame, self.frames_per_window))
            end_frame = max(0, min(end_frame, self.frames_per_window))

            # Позначаємо активність конкретного спікера [фрейми, індекс_спікера]
            labels[start_frame:end_frame, spk_idx] = 1.0

        return labels

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]

        # 1. Завантаження потрібного шматочка аудіо
        frame_offset = int(chunk['start_time'] * self.sample_rate)
        waveform, sr = torchaudio.load(
            chunk['audio_path'],
            frame_offset=frame_offset,
            num_frames=self.window_samples
        )

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Зводимо до моно і робимо тензор одновимірним (Samples,)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Паддинг на випадок, якщо кінець файлу завантажився некоректно або обірвався
        if waveform.shape[0] < self.window_samples:
            waveform = F.pad(waveform, (0, self.window_samples - waveform.shape[0]))

        # 2. Генерація мультилейбл міток для чанка (Frames, Max_Speakers)
        labels = self._generate_labels(
            chunk['start_time'],
            chunk['speech_segments'],
            chunk['speaker_to_idx']
        )

        return waveform, labels


class AMIDataModule(pl.LightningDataModule):
    def __init__(self, audio_dir, train_rttm_dir, val_rttm_dir,
                 window_size=2.0, window_hop=1.0, max_speakers=4,
                 batch_size=64, num_workers=4):
        super().__init__()
        self.audio_dir = audio_dir
        self.train_rttm_dir = train_rttm_dir
        self.val_rttm_dir = val_rttm_dir
        self.window_size = window_size
        self.window_hop = window_hop
        self.max_speakers = max_speakers
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = AMISegmentationDataset(
            audio_dir=self.audio_dir,
            rttm_dir=self.train_rttm_dir,
            window_size=self.window_size,
            window_hop=self.window_hop,
            max_speakers=self.max_speakers
        )
        self.val_dataset = AMISegmentationDataset(
            audio_dir=self.audio_dir,
            rttm_dir=self.val_rttm_dir,
            window_size=self.window_size,
            window_hop=self.window_hop,
            max_speakers=self.max_speakers
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools



class PermutationInvariantBCELoss(nn.Module):
    def __init__(self, num_speakers: int = 4):
        super().__init__()
        self.num_speakers = num_speakers
        self.permutations = list(itertools.permutations(range(num_speakers)))

    def forward(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds:   (Batch, Frames, Num_Speakers) - Logits
        targets: (Batch, Frames, Num_Speakers) - Ground truth binary labels
        """
        batch_size = preds.size(0)
        perm_losses = []

        # 1. Compute loss for all possible permutations
        for perm in self.permutations:
            perm_target = targets[:, :, list(perm)]
            loss = F.binary_cross_entropy_with_logits(preds, perm_target, reduction='none')
            loss_per_sample = loss.mean(dim=(1, 2))  # (Batch,)
            perm_losses.append(loss_per_sample)

        perm_losses = torch.stack(perm_losses, dim=0)  # (Num_Permutations, Batch)

        # 2. Find the minimum loss and the index of the best permutation for each sample
        min_losses, best_perm_indices = torch.min(perm_losses, dim=0)  # (Batch,)

        # 3. Vectorized target alignment based on the best permutation
        stacked_targets = torch.stack([targets[:, :, list(p)] for p in self.permutations], dim=0)  # (Num_Perm, B, T, C)

        # Expand indices to match stacked_targets dimensions for torch.gather
        gather_idx = best_perm_indices.view(1, batch_size, 1, 1).expand(
            1, batch_size, targets.size(1), targets.size(2)
        )
        aligned_targets = torch.gather(stacked_targets, dim=0, index=gather_idx).squeeze(0)

        return min_losses.mean(), aligned_targets


import torchaudio.transforms as T


class SpeakerSegmentationModel(nn.Module):
    def __init__(self, num_speakers: int = 4, sample_rate: int = 16000):
        super().__init__()

        # 10ms hop length gives exactly 100 frames per second
        self.spec_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )
        # Converts power spectrogram to decibel (log) scale
        self.amp_to_db = T.AmplitudeToDB()

        # Target only the FREQUENCY axis for pooling, leave TIME untouched
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # 80 mels -> 40 mels

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # 40 mels -> 20 mels
        )

        # 128 channels * 20 remaining frequency bins = 2560 features per time frame
        hidden_dim = 256
        self.lstm_proj = nn.Linear(128 * 20, hidden_dim)

        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3  # Prevents recurrent overfitting
        )

        # Final classification layer with dropout protection
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, num_speakers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (Batch, Samples) -> e.g., (Batch, 32000)
        """
        # 1. Generate Log-Mel Spectrogram -> (Batch, 1, Freqs, Frames)
        x = self.spec_transform(x)
        x = self.amp_to_db(x).unsqueeze(1)

        # 2. Extract spatial-frequency features
        x = self.feature_extractor(x)  # Shape: (Batch, 128, 20, Frames)

        # 3. Reshape seamlessly for sequence modeling
        batch, channels, freqs, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # Move 'Frames' (Time) to dimension 1
        x = x.view(batch, frames, channels * freqs)

        # 4. Temporal modeling
        x = F.relu(self.lstm_proj(x))
        x, _ = self.rnn(x)  # Shape: (Batch, Frames, Hidden_Dim * 2)

        # 5. Frame-level logits -> (Batch, Frames, Num_Speakers)
        logits = self.classifier(x)
        return logits


import pytorch_lightning as pl


class AMISegmentationTask(pl.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.num_speakers = model.classifier[-1].out_features

        self.loss_fn = PermutationInvariantBCELoss(num_speakers=self.num_speakers)

        # Set up TorchMetrics Collections for Train and Validation sets
        metrics = torchmetrics.MetricCollection([
            BinaryF1Score(),
            BinaryPrecision(),
            BinaryRecall(),
            BinaryAccuracy()
        ])

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        waveforms, targets = batch
        preds = self(waveforms)

        # Adjust frame dimensions if necessary due to CNN downsampling
        if preds.size(1) != targets.size(1):
            preds = F.interpolate(preds.permute(0, 2, 1), size=targets.size(1), mode='linear').permute(0, 2, 1)

        # Compute PIT loss and get aligned targets
        loss, aligned_targets = self.loss_fn(preds, targets)

        # Apply sigmoid to logits to get probabilities for metric evaluation
        preds_prob = torch.sigmoid(preds)

        # Flatten tensors for frame-level binary classification metrics
        preds_flat = preds_prob.reshape(-1)
        targets_flat = aligned_targets.reshape(-1)

        # Update and compute metrics based on the current stage
        if stage == "train":
            output_metrics = self.train_metrics(preds_flat, targets_flat)
        else:
            output_metrics = self.val_metrics(preds_flat, targets_flat)

        # Log metrics
        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(output_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }

def main():
    # Налаштування гіперпараметрів
    NUM_SPEAKERS = 4
    SAMPLE_RATE = 16000
    BATCH_SIZE = 16

    # Створюємо мок-датасети для демонстрації працездатності коду
    AUDIO_DIR = './pyannote/amicorpus'
    TRAIN_RTTM_DIR = './only_words/rttms/train'
    VAL_RTTM_DIR = './only_words/rttms/dev'

    data_module = AMIDataModule(audio_dir=AUDIO_DIR, train_rttm_dir=TRAIN_RTTM_DIR, val_rttm_dir=VAL_RTTM_DIR,
                                batch_size=32)

    # Ініціалізація моделі та Lightning модуля
    base_model = SpeakerSegmentationModel(num_speakers=NUM_SPEAKERS, sample_rate=SAMPLE_RATE)
    segmentation_task = AMISegmentationTask(model=base_model, lr=1e-3)

    # Налаштування тренера
    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="cuda",  # Автоматично обере CUDA або CPU
        devices=1,
        gradient_clip_val=5.0  # Захист від вибуху градієнтів в LSTM
    )

    # Запуск процесу навчання
    trainer.fit(segmentation_task, datamodule=data_module)


if __name__ == "__main__":
    main()