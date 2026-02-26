import os
import cv2
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .utils import overlay_mask

logger = logging.getLogger(__name__)

def inference_single_image(model, image_path, device, transform, target_size=(256, 256), 
                          threshold=0.5, mean=None, std=None):
    """
    Инференс для одного изображения (из ноутбука)
    
    Args:
        model: обученная модель
        image_path: путь к изображению или numpy array
        device: устройство (cuda/cpu)
        target_size: размер для ресайза (height, width)
        threshold: порог для бинаризации маски
        mean: среднее для нормализации
        std: стандартное отклонение для нормализации
    
    Returns:
        original_image: исходное изображение (RGB)
        probability_mask: маска вероятностей [0, 1]
        binary_mask: бинарная маска (0 или 1)
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]
    
    model.eval()
    
    # Загрузка изображения
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_path, np.ndarray):
        if image_path.shape[-1] == 3:
            image_rgb = image_path
        else:
            image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("image_path должен быть строкой или numpy array")
    
    original_h, original_w = image_rgb.shape[:2]
    
    # Применяем трансформации
    augmented = transform(image=image_rgb)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Инференс
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Возвращаем к исходному размеру
    probability_mask = cv2.resize(probability, (original_w, original_h), 
                                  interpolation=cv2.INTER_LINEAR)
    
    # Бинаризация
    binary_mask = (probability_mask > threshold).astype(np.uint8)
    
    return image_rgb, probability_mask, binary_mask

def process_folder(input_folder, output_folder, model, device, config):
    """
    Обработка всех изображений в папке
    
    Args:
        input_folder: путь к папке с изображениями
        output_folder: путь для сохранения результатов
        model: обученная модель
        device: устройство
        config: конфигурация с параметрами
    """
    
    # Создаем выходную папку, если её нет
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "prob_masks"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "overlay_mask"), exist_ok=True)
    
    # Поддерживаемые форматы изображений
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    # Собираем все изображения
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_folder).glob(f'*{ext}'))
        image_files.extend(Path(input_folder).glob(f'*{ext.upper()}'))
    
    if not image_files:
        logger.warning(f"В папке {input_folder} не найдено изображений")
        return
    
    logger.info(f"Найдено {len(image_files)} изображений для обработки")
    
    target_size = tuple(config['inference']['target_size'])
    threshold = config['inference']['threshold']
    mean = config['preprocessing']['mean']
    std = config['preprocessing']['std']
    
    # Обрабатываем каждое изображение
    successful = 0
    failed = 0
    
    # Трансформация для инференса
    transform = A.Compose([
        A.Resize(target_size[0], target_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    for img_path in tqdm(image_files, desc="Обработка изображений"):
        try:
            # Инференс
            original, prob_mask, binary_mask = inference_single_image(
                model=model,
                image_path=str(img_path),
                transform=transform,
                device=device,
                target_size=target_size,
                threshold=threshold,
                mean=mean,
                std=std
            )
            
            # Формируем имена выходных файлов
            stem = img_path.stem
            
            mask_path = os.path.join(output_folder, "masks", f"{stem}_mask.png")
            prob_mask_path = os.path.join(output_folder, "prob_masks", f"{stem}_prob_mask.png")
            overlay_path = os.path.join(output_folder, "overlay_mask",f"{stem}_overlay.png")
            
            # Сохраняем бинарную маску (0/255)
            cv2.imwrite(mask_path, (binary_mask * 255).astype(np.uint8))
            cv2.imwrite(prob_mask_path, (prob_mask * 255).astype(np.uint8))
            
            # Создаем и сохраняем наложение
            overlay = overlay_mask(
                cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
                binary_mask * 255,
                color=(0, 0, 255),
                alpha=0.6
            )
            cv2.imwrite(overlay_path, overlay)
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {img_path}: {e}")
            failed += 1
    
    logger.info(f"Обработка завершена. Успешно: {successful}, Ошибок: {failed}")