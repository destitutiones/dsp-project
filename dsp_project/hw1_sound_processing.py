import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
from scipy.signal import convolve, deconvolve

from dsp_project.utils import plot_input_tracks

dsp_project_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = dsp_project_dir_path.parent
sys.path.append(str(dsp_project_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.hw1.config import (  # noqa: E402 isort: skip
    SoundFilePaths,
    SoundProcessingParams,
)


def process_sound(
    cfg_proc_params: SoundProcessingParams,
    cfg_filepaths: SoundFilePaths,
    original: List[float],
    record: List[float],
) -> None:
    """
    Реализация основной части дз №1.

    :param cfg_proc_params: конфиг с характеристиками аудиодорожек и параметрами алгоритма обработки
    :param cfg_filepaths: конфиг с путями до обрабатываемых данных
    :param original: оригинальная запись свипера
    :param record: запись свипера с микрофона
    :return: None
    """

    band_cnt = cfg_proc_params["band_cnt"]
    sampling_rate = cfg_proc_params["sampling_rate"]

    # 2. Переводим оригинальный свипер и записанный свипер в частотную область
    original_fft = np.abs(np.fft.rfft(original))
    record_fft = np.abs(np.fft.rfft(record))
    sweeper_freq_domain = np.fft.rfftfreq(len(original), 1 / sampling_rate)

    plot_input_tracks(
        {"original": original_fft, "record": record_fft},
        "Plotting input audio tracks fft-transformation graph",
        "Discrete Fourier Transform",
        "Frequency",
        "Amplitude",
        sweeper_freq_domain,
    )
    # Видим, что набор полученных частот sweeper_freq_domain совпадает с действительным,
    # поэтому дополнительных корректировок не производим

    # 3. Бьем частотную область на полосы (aka бэнды будущего эквалайзера)
    # Рассматриваем только значения, попадающие в допустимый промежуток
    def make_bands(values):
        return np.array_split(values, band_cnt)

    original_fft_bands = make_bands(original_fft)
    record_fft_bands = make_bands(record_fft)
    freq_bands = make_bands(sweeper_freq_domain)

    # 4. В каждой полосе берем среднее значение амплитуды
    def get_mean_amplitude_for_band(band_list):
        return np.fromiter(map(lambda x: np.mean(x), band_list), float)

    original_band_mean = get_mean_amplitude_for_band(original_fft_bands)
    record_band_mean = get_mean_amplitude_for_band(record_fft_bands)

    # 6. Делим набор значений для оригинального свипера на набор значений
    # для записанного – получаем набор гейнов эквалайзера
    equalizer_gain = original_band_mean / record_band_mean
    # Убираем эффект громкости дорожки, нормируя на средний гейн
    equalizer_gain /= equalizer_gain.mean()

    # 7. Аналогично обрабатываем файл с белым шумом

    # Пути до данных
    data_dir_path = cfg_filepaths["data_dir_path"]
    noise_white_file_name = cfg_filepaths["noise_white_file_name"]
    noise_white_record_file_name = cfg_filepaths["noise_white_record_file_name"]
    noise_white_adjusted_file_name = cfg_filepaths["noise_white_adjusted_file_name"]

    logging.info("Loading white noise file")
    white_noise, _ = sf.read(data_dir_path + noise_white_file_name)
    white_noise = white_noise[:, 0]
    white_noise_fft = np.abs(np.fft.rfft(white_noise))
    white_noise_freq_domain = np.fft.rfftfreq(len(white_noise), 1 / sampling_rate)

    # Корректируем АЧХ колонки, умножая всю i-ую полосу на i-ый полученный гейн
    # Так как границы частот полос могут отличаться, не используем метод make_bands,
    # а проходим по значениям white_noise_freq_domain и сопоставляем их полосам freq_bands
    white_noise_fft_adjusted = white_noise_fft.copy()
    for i in range(band_cnt):
        idx = np.where(
            (white_noise_freq_domain >= freq_bands[i][0])
            & (white_noise_freq_domain <= freq_bands[i][-1])
        )
        white_noise_fft_adjusted[idx] *= equalizer_gain[i]

    plot_input_tracks(
        {
            "white_noise": white_noise_fft,
            "white_noise_adjusted": white_noise_fft_adjusted,
        },
        "Plotting white noise fft-transformation graph",
        "Discrete Fourier Transform",
        "Frequency",
        "Amplitude",
        white_noise_freq_domain,
    )
    # Судя по графику white_noise_adjusted, на высоких частотах расхождение записи
    # с оригиналом очень высокое, на более низких приемлемое

    # 8. Скорректированный файл возвращаем во временную область
    white_noise_adjusted = np.fft.irfft(white_noise_fft_adjusted)
    logging.info("Saving adjusted white noise file")
    sf.write(
        data_dir_path + noise_white_adjusted_file_name,
        white_noise_adjusted,
        sampling_rate,
    )

    # 10. Считаем деконволюцию от записанного шума и оригинального, получаем импульсный отклик
    white_noise_record, _ = sf.read(data_dir_path + noise_white_record_file_name)
    white_noise_deconv = deconvolve(white_noise_record, white_noise)

    # Обработка тестового файла
    # Пути до данных
    test_file_name = cfg_filepaths["test_file_name"]
    test_file_record_name = cfg_filepaths["test_file_record_name"]
    test_file_record_conv_name = cfg_filepaths["test_file_record_conv_name"]

    test_file, _ = sf.read(data_dir_path + test_file_name)
    test_record_file, _ = sf.read(data_dir_path + test_file_record_name)

    # 2. Сравним этот оригинал с полученным в эксперименте импульсным откликом
    test_record_file_conv = convolve(test_record_file, white_noise_deconv[0])
    plot_input_tracks(
        {
            "test_record": test_record_file,
            "test_record_w_impulse": test_record_file_conv,
        },
        "Plotting test file comparison graph",
        "Test records",
        "Time",
        "Amplitude",
    )
    logging.info("Saving test file with impulse")
    sf.write(
        data_dir_path + test_file_record_conv_name, test_record_file_conv, sampling_rate
    )
    # Запись получилась достаточно тихой, но стала менее зашумленной (или я просто перестала слышать шум)
