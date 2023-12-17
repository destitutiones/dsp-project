import logging
import os
import subprocess
from typing import Dict, List, Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from torchmetrics.audio import (  # isort: skip
    PerceptualEvaluationSpeechQuality,
    ScaleInvariantSignalDistortionRatio,
    SignalDistortionRatio,
)


def download_file_from_gdrive(gdrive_file_id: str, outfile: str) -> None:
    """
    Скачиваем файлы из Google Drive по ID.

    :param gdrive_file_id: id файла на гугл-диске
    :param outfile: путь, по которому будет сохранен файл из гугл-диска
    :return: None
    """
    outfile_dir = outfile[: outfile.rfind("/") + 1]

    if not (len(outfile_dir) == "" or os.path.exists(outfile_dir)):
        logging.info(f"Creating directory {outfile_dir}")
        os.makedirs(outfile_dir)

    upload_cmd = (
        "wget --load-cookies /tmp/cookies.txt"
        ' "https://docs.google.com/uc?export=download&confirm=$('
        " wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies"
        f" --no-check-certificate 'https://docs.google.com/uc?export=download&id={gdrive_file_id}'"
        f" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={gdrive_file_id}\" "
        f" -O {outfile} && rm -rf /tmp/cookies.txt"
    )
    logging.info(f"Downloading zip-archive to {outfile} file")
    subprocess.check_call(upload_cmd, shell=True)
    logging.info("Download complete")


def get_gdrive_file_id(link: str) -> str:
    """
    Получение id файла из url'a.
    :param link: адрес файла
    :return: id файла
    """
    return link.split("/")[5]


def unzip_archive(zip_file_path: str, output_files_path: str) -> None:
    """
    Извлечение файлов из архива в указанную папку.

    :param zip_file_path: путь до архива
    :param output_files_path: путь до папки для извлечения файлов
    :return: None
    """
    from zipfile import ZipFile

    # Если указанного пути для извлечения файлов не существует,
    # создаем папку с заданным названием
    if not os.path.exists(output_files_path):
        logging.info(f"Creating directory {output_files_path}")
        os.makedirs(output_files_path)

    with ZipFile(zip_file_path, "r") as zip_ref:
        logging.info(f"Extracting files from {zip_file_path}")
        zip_ref.extractall(output_files_path)
        logging.info("Extracting complete")


def prepare_data(archive_gdrive_url, output_zip_path, output_files_path) -> None:
    """
    Скачиваем архив с input-файлами из гугл-диска и распаковываем архив.

    :param archive_gdrive_url: ссылка на архив с данными, гугл-диск
    :param output_zip_path: путь, по которому будет сохранен архив из гугл-диска
    :param output_files_path: путь до папки для извлечения файлов
    :return: None
    """
    # Парсинг url'a для получения id файла на диске
    file_id = get_gdrive_file_id(archive_gdrive_url)
    # Загрузка файла с гугл-диска
    download_file_from_gdrive(file_id, output_zip_path)
    # Распаковка архива
    unzip_archive(output_zip_path, output_files_path)
    logging.info("Data prepared for work")


def plot_input_tracks(
    values: Dict[str, List[float]],
    log_message: str,
    title: str,
    xlabel: str,
    ylabel: str,
    frequencies: Optional[List[float]] = None,
) -> None:
    """
    Отрисовываем полученные на ввод значения, соответствующие оригиналу и записи.
    Максимум 5 линий.

    :param values: словарь лейбл: значения графика
    :param log_message: сообщение, передаваемое в логгер
    :param title: заголовок графика
    :param xlabel: значение оси абсцисс
    :param ylabel: значение оси ординат
    :param frequencies: значения частот в случае, если рисуем fft
    :return: None
    """

    sns.set_style("whitegrid")
    pallette = ["royalblue", "hotpink", "limegreen", "gold", "tomato"]

    plt.figure(figsize=(12, 6))
    logging.info(log_message)
    for i, (k, v) in enumerate(values.items()):
        plt.plot(
            np.arange(len(v)) if frequencies is None else frequencies,
            v,
            color=pallette[i],
            alpha=0.8,
            label=k,
        )
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def prepare_noise(noise: List[float], original: List[float]) -> List[float]:
    """
    - Обрезаем шум, если он короче оригинала, в противном случае стекаем его, пока
    длина аудиодорожки не станет равна длине оригинала.
    - Выравниваем громкость шума по оригиналу.
    :param noise: запись шума
    :param original: запись оригинальной аудиодорожки
    :return:
    """
    noise_len = len(noise)
    original_len = len(original)

    if original_len < noise_len:
        noise = noise[
            np.tile(np.arange(noise_len), int(np.ceil(original_len / noise_len)))
        ]

    noise = noise[:original_len]

    P_original = np.mean(original**2)
    P_noise = np.mean(noise**2)

    # Выровняем громкость шума
    noise *= np.sqrt(P_original / P_noise)

    return noise


def calculate_snr(snr_db: float) -> float:
    """
    Считаем SNR (signal-to-noise ratio) по значению SNR в децибелах.

    Пользуемся следующими уравнениями:
    - snr_db = 10*log_10(snr)
    - A_s / A_n = sqrt(snr)

    :param snr_db: значение signal-to-noise ratio в децибелах
    :return: значение snr = A_signal / A_noise
    """

    snr = 10 ** (snr_db / 20)
    return snr


def calculate_metrics(
    original: List[float],
    mix: List[float],
    sampling_rate,
    pesq_sampling_rate,
    pesq_mode,
) -> Dict:
    """
    Считаем перечисленные в задании метрики.
    :param original: запись чистого голоса
    :param mix: смесь чистого голоса и шума
    :param sampling_rate: sr исходной аудиозаписи
    :param pesq_sampling_rate: параметр fs для расчета Perceptual Evaluation Speech Quality
    :param pesq_mode: параметр mode для расчета Perceptual Evaluation Speech Quality
    :return: pd.DataFrame с метриками
    """

    # Остальные метрики считаем в юпитер-ноутбуке, т.к. установка в poetry оказалась затруднительной
    METRICS_TO_CALC = {
        "PESQ": [PerceptualEvaluationSpeechQuality, (pesq_sampling_rate, pesq_mode)],
        "SI-SDR": [ScaleInvariantSignalDistortionRatio, ()],
        "SDR": [SignalDistortionRatio, ()],
    }

    def make_torch(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if len(x.shape) > 1:
            x = x[0]
        return x

    torch_original = make_torch(original)
    torch_mix = make_torch(mix)

    torch_original_resampled = make_torch(
        librosa.resample(
            np.array(original), orig_sr=sampling_rate, target_sr=pesq_sampling_rate
        )
    )
    torch_mix_resampled = make_torch(
        librosa.resample(
            np.array(mix), orig_sr=sampling_rate, target_sr=pesq_sampling_rate
        )
    )

    metrics = {}

    for name, (module, params) in METRICS_TO_CALC.items():
        module_obj = module(*params)
        if name == "PESQ":
            metrics_val = float(
                module_obj(torch_original_resampled, torch_mix_resampled)
            )
        else:
            metrics_val = float(module_obj(torch_original, torch_mix))
        metrics[name] = metrics_val

    return metrics
