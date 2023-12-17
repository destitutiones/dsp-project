import logging
import sys
from pathlib import Path
from typing import List

import hydra
import pandas as pd
import soundfile as sf

from dsp_project.utils import (  # isort: skip
    calculate_metrics,
    calculate_snr,
    prepare_data,
    prepare_noise,
)

dsp_project_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = dsp_project_dir_path.parent
sys.path.append(str(dsp_project_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.hw1.config import Params  # noqa: E402


def mixer(original: List[float], noise: List[float], snr_db: int) -> List[float]:
    """
    Смешиваем чистый голос и шум по заданному в децибелах SNR.

    :param original: запись чистого голоса
    :param noise: запись шума
    :param snr_db: значение signal-to-noise ratio
    :return: смесь чистого голоса и шума
    """
    # Подгоняем шум под длину и громкость записи голоса
    noise = prepare_noise(noise, original)

    snr = calculate_snr(snr_db)
    original_coeff = snr / (1 + snr)

    mix = original_coeff * original + (1 - original_coeff) * noise

    return mix


@hydra.main(config_path="../configs/hw2", config_name="config", version_base="1.3")
def main(cfg: Params):
    # Все пути, константы и параметры запуска берем из конфига
    # Скачивание аудиодорожек из гугл-диска
    if_download_files = cfg["gdrive_params"]["if_download_files"]
    gdrive_sound_zip_path = cfg["gdrive_params"]["gdrive_sound_zip_path"]
    # Пути до данных
    data_dir_path = cfg["sound_files"]["data_dir_path"]
    zip_file_name = cfg["sound_files"]["zip_file_name"]
    noise_file_name = cfg["sound_files"]["noise_file_name"]
    speech_file_name = cfg["sound_files"]["speech_file_name"]
    # Параметры смешивания
    snr_db_vals = cfg["sound_proc_params"]["snr_db"]
    sampling_rate = cfg["sound_proc_params"]["sampling_rate"]
    # Параметры для расчета метрик
    pesq_sampling_rate = cfg["sound_proc_params"]["pesq_sampling_rate"]
    pesq_mode = cfg["sound_proc_params"]["pesq_mode"]

    if if_download_files:
        prepare_data(
            gdrive_sound_zip_path, data_dir_path + zip_file_name, data_dir_path
        )

    logging.info("Loading input files")
    speech, _ = sf.read(data_dir_path + speech_file_name)
    noise, _ = sf.read(data_dir_path + noise_file_name)
    noise = noise[:, 0]

    metrics_list = []

    for snr_db in [None] + snr_db_vals:
        if snr_db is None:
            mix = speech
            mix_file_name = speech_file_name
        else:
            logging.info(f"Mixing speech and noise with snr={snr_db}dB")
            mix = mixer(speech, noise, snr_db)
            mix_file_name = f"mixed_{snr_db}.wav"
            sf.write(data_dir_path + mix_file_name, mix, sampling_rate)

        metrics = calculate_metrics(
            speech, mix, sampling_rate, pesq_sampling_rate, pesq_mode
        )
        metrics["file"] = mix_file_name
        metrics["SNR, dB"] = snr_db
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    print(metrics_df[["file", "SNR, dB", "SDR", "SI-SDR", "PESQ"]])


if __name__ == "__main__":
    main()
