import os
import logging
import torch
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)

def load_model(model_name, device, weights_dir='weights'):
    """
    Загрузка модели сегментации по имени с весами из указанной директории
    
    Args:
        model_name: название модели ('unet_resnet34', 'deeplabv3plus_resnet50', 'unetplusplus_efficientnetb3')
        device: устройство для загрузки ('cuda' или 'cpu')
        weights_dir: директория с файлами весов
    
    Returns:
        model: загруженная модель в режиме eval()
    """
    
    # Маппинг названий моделей на классы SMP
    model_classes = {
        'unet_resnet34': {
            'class': smp.Unet,
            'encoder': 'resnet34',
            'params': {}
        },
        'deeplabv3plus_resnet50': {
            'class': smp.DeepLabV3Plus,
            'encoder': 'resnet50',
            'params': {}
        },
        'unetplusplus_efficientnetb3': {
            'class': smp.UnetPlusPlus,
            'encoder': 'efficientnet-b3',
            'params': {}
        }
    }
    
    if model_name not in model_classes:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(model_classes.keys())}")
    
    # Путь к файлу весов
    weights_path = os.path.join(weights_dir, f"{model_name}.pth")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Файл весов не найден: {weights_path}")
    
    logger.info(f"Загрузка модели {model_name} с весами из {weights_path}")
    
    # Инициализация архитектуры модели (без предобученных весов)
    model_info = model_classes[model_name]
    model = model_info['class'](
        encoder_name=model_info['encoder'],
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    
    # Загрузка весов
    try:
        checkpoint = torch.load(weights_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Загружена модель с IoU: {checkpoint.get('val_iou', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Веса загружены успешно")
            
    except Exception as e:
        logger.error(f"Ошибка при загрузке весов: {e}")
        raise
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Модель {model_name} успешно загружена на {device}")
    return model