#!/usr/bin/env python3
"""
Демо-скрипт для инференса модели сегментации огня
Запуск: python demo.py --input /path/to/images --output ./results
"""

import os
import sys
import argparse
import logging
import yaml
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import load_model, process_folder, setup_logging

logger = logging.getLogger(__name__)

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Инференс модели сегментации огня",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Путь к папке с входными изображениями'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Путь для сохранения результатов'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Название модели для инференса (по умолчанию из конфига)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/inference.yaml',
        help='Путь к конфигурационному файлу'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default=None,
        help='Устройство для инференса (по умолчанию из конфига)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Подробный вывод'
    )
    
    return parser.parse_args()


def load_config(config_path):
    """Загрузка конфигурации из YAML файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Конфигурация загружена из {config_path}")
        return config
    except Exception as e:
        logger.error(f"Ошибка при загрузке конфига: {e}")
        sys.exit(1)


def main():
    """Основная функция"""
    args = parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    config = load_config(args.config)
    
    if args.device:
        device = args.device
    else:
        device = config['inference']['device']
    
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA не доступна, используется CPU")
        device = 'cpu'
    
    logger.info(f"Используется устройство: {device}")
    
    # Определяем модель для загрузки
    model_name = args.model if args.model else config['paths']['default_model']
    weights_dir = config['paths']['weights_dir']
    
    # Проверяем существование входной папки с картинками
    if not os.path.exists(args.input):
        logger.error(f"Входная папка не существует: {args.input}")
        sys.exit(1)
    
    try:
        logger.info(f"Загрузка модели: {model_name}")
        model = load_model(
            model_name=model_name,
            device=device,
            weights_dir=weights_dir
        )
        
        logger.info(f"Начало обработки папки: {args.input}")
        process_folder(
            input_folder=args.input,
            output_folder=args.output,
            model=model,
            device=device,
            config=config
        )
        
        logger.info(f"Результаты сохранены в: {args.output}")
        
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        logger.info("Убедитесь, что файлы весов находятся в папке weights/")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()