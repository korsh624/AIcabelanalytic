from ultralytics import YOLO
import torch
import time

# Проверяем доступность GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Настройки
DATA_YAML = "data.yaml"  # Путь к вашему data.yaml
MODEL = "yolo11x.pt"  # Загрузите предварительно обученную модель
BATCH_SIZE = 16  # Размер батча
EPOCHS = 299  # Количество эпох

# Загружаем модель YOLO
model = YOLO(MODEL)


# Коллбэк для отслеживания времени эпох
class EpochTimer:
    def init(self):
        self.start_time = None

    def on_train_epoch_start(self, trainer):
        self.start_time = time.time()
        current_epoch = trainer.epoch + 1
        print(f"\n[Эпоха {current_epoch}/{trainer.epochs}] началась")

    def on_train_epoch_end(self, trainer):
        elapsed_time = time.time() - self.start_time
        current_epoch = trainer.epoch + 1
        mAP = trainer.metrics.get("metrics/mAP50", 0)  # Получение mAP из метрик
        print(f"[Эпоха {current_epoch} завершена] | mAP: {mAP:.4f} | Время: {elapsed_time:.2f} сек")


# Экземпляр коллбэка
epoch_timer = EpochTimer()

# Подключение коллбэков
model.add_callback('on_train_epoch_start', epoch_timer.on_train_epoch_start)
model.add_callback('on_train_epoch_end', epoch_timer.on_train_epoch_end)

# Запуск обучения
model.train(
    data=DATA_YAML,  # Датасет
    epochs=EPOCHS,  # Количество эпох
    imgsz=640,  # Размер изображения
    batch=BATCH_SIZE,  # Размер батча
    device=device,  # Устройство: GPU или CPU
    workers=4,  # Воркеры для загрузки данных
    project='yolo_training',  # Папка для сохранения моделей
    name='gpu_experiment3',  # Название эксперимента
    save=True  # Сохраняем лучшую модель
)