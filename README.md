```markdown
# Fire Segmentation

Скрипт для инференса модели сегментации огня на основе обученных нейросетей.

## Описание

Данный инструмент принимает на вход папку с изображениями и генерирует маски интенсивности огня с использованием одной из трех предобученных моделей сегментации.

## Структура проекта

```
fire_segmentation_inference/
├── config/                             # Конфигурационные файлы
│   └── inference.yaml                  # Параметры инференса и пути
├── notebooks/                          # Директория с ноутбуками
│   └── train_models.ipynb              # Ноутбук с историей обучения и визуализации метрик моделей
├── weights/                            # Директория с весами моделей
│   ├── unet_resnet34.pth               # Веса U-Net с ResNet34
│   ├── deeplabv3plus_resnet50.pth      # Веса DeepLabV3+ с ResNet50
│   └── unetplusplus_efficientnetb3.pth # Веса U-Net++ с EfficientNet-B3
├── src/                                # Исходный код
│   ├── __init__.py
│   ├── model_loader.py                 # Загрузка архитектуры и весов
│   ├── inference_engine.py             # Ядро инференса
│   └── utils.py                        # Вспомогательные функции
├── demo.py                             # Главный скрипт (точка входа)
├── requirements.txt                    # Зависимости проекта
└── README.md                           # Документация
```

## Быстрый старт

### 1. Установка

```bash
# Клонирование репозитория
git clone https://github.com/egortens1a/fire-segmentation.git
cd fire_segmentation_inference

# Установка зависимостей
pip install -r requirements.txt
```

### 2. Подготовка весов

Проверьте наличие файлов с весами обученных моделей в папку `weights/` (достаточно наличия используемой модели):
- `unet_resnet34.pth` - базовая модель U-Net
- `deeplabv3plus_resnet50.pth` - DeepLabV3+ для сложных сцен
- `unetplusplus_efficientnetb3.pth` - U-Net++ для мелких деталей

### 3. Запуск инференса

```bash
# Базовый запуск (модель по умолчанию - unet_resnet34)
python demo.py --input /путь/к/изображениям --output /путь/для/результатов

# Пример с конкретными папками
python demo.py --input data/input/ --output data/output/

# Выбор конкретной модели
python demo.py --input data/input/ --output data/output/ --model deeplabv3plus_resnet50

# Использование CPU
python demo.py --input data/input/ --output data/output/ --device cpu

# Подробный вывод для отладки
python demo.py --input data/input/ --output data/output/ --verbose
```

## Входные данные

- **Формат изображений**: JPG, JPEG, PNG, BMP, TIFF
- **Цветовое пространство**: RGB (автоматически конвертируется)
- **Разрешение**: любое (модель работает с 256x256, маски возвращаются к исходному размеру)

## Выходные данные

Для каждого входного изображения `image.jpg` создаются 3 файла:

| Файл | Описание |
|------|----------|
| `{image_name}_mask.png` | Бинарная маска (0/255) - белым выделены области огня |
| `{image_name}_prob_mask.png` | Бинарная маска (0/255) - белым выделены области огня |
| `{image_name}_overlay.png` | Исходное изображение с наложенной полупрозрачной красной маской |

## Конфигурация

Все параметры инференса настраиваются в файле `config/inference.yaml`:

```yaml
inference:
  device: "cuda"                 # Устройство: "cuda" или "cpu"
  target_size: [256, 256]        # Размер для модели (height, width)
  threshold: 0.5                  # Порог бинаризации маски
  batch_size: 16                  # Размер батча

paths:
  weights_dir: "weights"          # Папка с весами
  default_model: "unet_resnet34"   # Модель по умолчанию

preprocessing: # параметры распределений, соотвествующие распределению цветов в ImageNet (энкодеры были предобучены на нем)
  mean: [0.485, 0.456, 0.406] 
  std: [0.229, 0.224, 0.225]
```

## Доступные модели

| Модель | Архитектура | Особенности |
|--------|-------------|-------------|
| `unet_resnet34` | U-Net с ResNet34 | Быстрый baseline, низкие требования к памяти |
| `deeplabv3plus_resnet50` | DeepLabV3+ с ResNet50 | Хорош для сложных сцен, точная сегментация |
| `unetplusplus_efficientnetb3` | U-Net++ с EfficientNet-B3 | Лучше передаёт многоуровневые признаки |

