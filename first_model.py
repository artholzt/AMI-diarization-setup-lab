import os
import torch
import torchaudio
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard.summary import audio
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from pathlib import Path


class AMIVADDataset(Dataset):
    def __init__(self, audio_dir, rttm_dir, window_size=2.0, window_hop=1.0,
                 sample_rate=16000, feature_hop_ms=10):
        """
        Args:
            audio_dir: Шлях до папки з wav файлами.
            rttm_dir: Шлях до папки з rttm файлами.
            window_size: Розмір аудіо-чанка в секундах (напр. 2.0с).
            window_hop: Крок зсуву вікна при нарізці довгих аудіо (напр. 1.0с для перекриття).
            sample_rate: Частота дискретизації.
            feature_hop_ms: Крок вікна спектрограми в мілісекундах (має збігатися з моделлю).
        """
        self.audio_dir = Path(audio_dir)
        self.rttm_dir = Path(rttm_dir)
        self.window_size = window_size
        self.window_hop = window_hop
        self.sample_rate = sample_rate

        # Обчислюємо скільки фреймів (міток) буде в одному вікні
        # Наприклад, 2.0 сек / 0.01 сек (10мс) = 200 фреймів
        self.frames_per_window = int(window_size * 1000 / feature_hop_ms)
        self.feature_hop_sec = feature_hop_ms / 1000.0
        self.window_samples = int(window_size * sample_rate)

        self.chunks = self._prepare_data()

    def _parse_rttm(self, rttm_path):
        """Парсить RTTM файл і повертає об'єднані інтервали мовлення."""
        segments = []
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Формат RTTM: SPEAKER file_id channel start_time duration ...
                if len(parts) >= 5 and parts[0] == "SPEAKER":
                    start = float(parts[3])
                    duration = float(parts[4])
                    segments.append((start, start + duration))

        # Об'єднання (merge) інтервалів, що перекриваються
        if not segments:
            return []

        segments.sort(key=lambda x: x[0])
        merged_segments = [segments[0]]

        for current_start, current_end in segments[1:]:
            prev_start, prev_end = merged_segments[-1]
            if current_start <= prev_end:
                # Якщо перекриваються, розширюємо попередній інтервал
                merged_segments[-1] = (prev_start, max(prev_end, current_end))
            else:
                # Якщо ні, додаємо новий
                merged_segments.append((current_start, current_end))

        return merged_segments

    def _prepare_data(self):
        """Сканує файли та нарізає їх на логічні чанки."""
        chunks = []

        # Шукаємо всі wav файли
        audio_files = list(self.audio_dir.glob('*.wav'))

        # for audio_path in audio_files:
        for audio_path in audio_files:
            file_id = audio_path.stem.split('.')[0]
            rttm_path = self.rttm_dir / f"{file_id}.rttm"

            if not rttm_path.exists():
                continue

            # Отримуємо інтервали мовлення
            speech_segments = self._parse_rttm(rttm_path)

            # Отримуємо загальну тривалість аудіо без завантаження всього файлу в RAM
            info = torchaudio.info(audio_path)
            total_duration = info.num_frames / info.sample_rate

            # Нарізаємо довге аудіо на чанки (наприклад, по 2 секунди)
            current_time = 0.0
            while current_time + self.window_size <= total_duration:
                # Зберігаємо інформацію про чанк, щоб завантажити його пізніше у __getitem__
                chunks.append({
                    'audio_path': audio_path,
                    'start_time': current_time,
                    'speech_segments': speech_segments  # передаємо всі сегменти, відфільтруємо в getitem
                })
                current_time += self.window_hop

        return chunks

    def _generate_labels(self, chunk_start, speech_segments):
        """Генерує вектор нулів та одиниць для фреймів чанка."""
        labels = torch.zeros(self.frames_per_window)
        chunk_end = chunk_start + self.window_size

        for seg_start, seg_end in speech_segments:
            # Якщо сегмент мовлення взагалі не перетинається з нашим чанком — пропускаємо
            if seg_end <= chunk_start or seg_start >= chunk_end:
                continue

            # Визначаємо, які саме фрейми в межах нашого чанка містять мову
            # Обрізаємо межі сегмента до меж чанка
            overlap_start = max(chunk_start, seg_start)
            overlap_end = min(chunk_end, seg_end)

            # Переводимо час (секунди) в індекси фреймів
            start_frame = int((overlap_start - chunk_start) / self.feature_hop_sec)
            end_frame = int((overlap_end - chunk_start) / self.feature_hop_sec)

            # Ставимо 1 (мова) для відповідних фреймів
            labels[start_frame:end_frame] = 1.0

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

        # Перевірка частоти дискретизації (якщо потрібно - робимо ресемплінг)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # Якщо в аудіо декілька каналів (стерео) - зводимо до моно
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. Генерація бінарних міток для цього чанка
        labels = self._generate_labels(chunk['start_time'], chunk['speech_segments'])

        return waveform, labels


class AMIDataModule(pl.LightningDataModule):
    def __init__(self, audio_dir, train_rttm_dir, val_rttm_dir, batch_size=128, num_workers=0):
        super().__init__()
        self.audio_dir = audio_dir
        self.train_rttm_dir = train_rttm_dir
        self.val_rttm_dir = val_rttm_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Розбиття на train/val/test згідно з протоколом AMI
        self.train_dataset = AMIVADDataset(self.audio_dir, self.train_rttm_dir)
        self.val_dataset = AMIVADDataset(self.audio_dir, self.val_rttm_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class VAD_CRNN(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=64):
        super().__init__()

        # Витягування ознак: Mel-спектрограма
        self.feature_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,  # 10ms фрейми
            n_mels=n_mels
        )

        # Згортковий блок для локальних ознак
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Рекурентний блок для часового контексту
        self.rnn = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)

        # Класифікатор
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),  # 64 * 2 (бо bidirectional)
            nn.ReLU(),
            nn.Linear(64, 1)  # Вихід: логіти для BCEWithLogitsLoss
        )

    def forward(self, x):
        # x shape: [batch, channels, time]
        x = self.feature_extractor(x)
        # x shape: [batch, n_mels, frames]
        x = x.squeeze(1) if x.dim() == 4 else x  # Видаляємо зайвий вимір каналів, якщо є

        # Аммплітуда в логарифмічний масштаб
        x = torch.log(x + 1e-6)

        x = self.cnn(x)
        # Транспонування для RNN: [batch, channels, time] -> [batch, time, channels]
        x = x.transpose(1, 2)

        rnn_out, _ = self.rnn(x)

        logits = self.classifier(rnn_out)
        return logits.squeeze(-1)  # [batch, time]


class VADLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = VAD_CRNN()
        self.criterion = nn.BCEWithLogitsLoss()

        # Метрики
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx):
        waveforms, labels = batch
        logits = self(waveforms)

        # Вирівнювання розмірностей (interpolating labels if needed)
        # Оскільки CNN з poolings зменшує часовий вимір, нам потрібно вирівняти логіти і мітки
        if logits.shape[1] != labels.shape[1]:
            logits = nn.functional.interpolate(logits.unsqueeze(1), size=labels.shape[1], mode='linear').squeeze(1)

        loss = self.criterion(logits, labels)
        preds = torch.sigmoid(logits) > 0.5

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.train_acc(preds, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch, batch_idx)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

if __name__=='__main__':
    # Шляхи до розпакованого датасету
    AUDIO_DIR = './pyannote/amicorpus'
    TRAIN_RTTM_DIR = './only_words/rttms/train'
    VAL_RTTM_DIR = './only_words/rttms/dev'

    data_module = AMIDataModule(audio_dir=AUDIO_DIR, train_rttm_dir=TRAIN_RTTM_DIR, val_rttm_dir=VAL_RTTM_DIR,
                                batch_size=32)

    model = VADLightningModule(learning_rate=1e-3)

    # Колбеки для збереження найкращої моделі та ранньої зупинки
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_f1',
        dirpath='checkpoints/',
        filename='vad-ami-{epoch:02d}-{val_f1:.2f}',
        save_top_k=3,
        mode='max',
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator='cuda',  # Автоматично використовує GPU (MPS для Mac, CUDA для Nvidia)
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        # callbacks=[checkpoint_callback,],
        precision='16-mixed',  # Mixed precision для швидкості
    )

    trainer.fit(model, datamodule=data_module)

