import cv2
import numpy as np
import logging
import sys
from pathlib import Path

def overlay_mask(img: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255), alpha:float=0.5):
    """
    Накладывает маску на изображение полупрозрачным цветом (из ноутбука)
    
    Параметры:
    - img: np.ndarray, uint8, shape (H,W,3) - входное изображение.
    - mask: np.ndarray, shape (H,W) - бинарная (0/255) или float (0..1).
    - color: tuple of 3 int (0-255) - цвет выделения.
    - alpha: float [0..1] - непрозрачность наложения.
    - img_is_bgr: bool - True, если img в BGR (OpenCV), False - если RGB.
    
    Возвращает:
    - out: np.ndarray, uint8, shape (H,W,3) - изображение с наложенной маской.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("img должен быть HxWx3 uint8")
    
    h, w = img.shape[:2]
    if mask.shape != (h, w):
        raise ValueError("mask должен иметь форму HxW, совпадающую с img")
    
    # приводим mask к float 0..1
    if mask.dtype == np.uint8:
        if mask.max() > 1:
            m = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
        else:
            m = mask.astype(np.float32).clip(0.0, 1.0)
    else:
        m = mask.astype(np.float32)
        # если значения вне 0..1, нормализуем
        if m.max() > 1.0 or m.min() < 0.0:
            m = (m - m.min()) / (m.max() - m.min() + 1e-9)
    
    # Создаем цветной слой того же размера
    color_arr = np.zeros_like(img, dtype=np.uint8)
    color_arr[:] = tuple(int(c) for c in color)
    
    # Преобразуем к float для смешивания
    img_f = img.astype(np.float32) / 255.0
    color_f = color_arr.astype(np.float32) / 255.0
    alpha = float(np.clip(alpha, 0.0, 1.0))
    
    # Расширяем маску до 3 каналов
    m3 = np.dstack([m] * 3)  # shape (H,W,3), float32
    
    # Итоговая непрозрачность для каждого пикселя = alpha * mask
    A = (alpha * m3).clip(0.0, 1.0)
    
    out_f = (1.0 - A) * img_f + A * color_f
    out = (out_f * 255.0).astype(np.uint8)
    
    return out


def setup_logging(level=logging.INFO):
    """Настройка логирования"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    
def ensure_dir(path):
    """Создание директории, если её нет"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path