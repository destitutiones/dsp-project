from dataclasses import dataclass
from typing import List


@dataclass
class GdriveParams:
    """
    Скачивание аудиодорожек из гугл-диска.

    if_download_files: нужно ли выгружать данные из гугл-диска
    gdrive_sound_zip_path: путь до zip-архива со всеми входными дорожками
    """

    if_download_files: bool
    gdrive_sound_zip_path: str


@dataclass
class SoundFilePaths:
    """
    Пути до обрабатываемых данных.

    data_dir_path: путь до папки, в которой расположен архив/куда он будет скачан
     и куда будут извлечены файлы
    zip_file_name: название архива с аудиодорожками
    noise_file_name: название файла с шумом
    speech_file_name: название файла с чистым голосом
    """

    data_dir_path: str
    zip_file_name: str
    speech_file_name: str


@dataclass
class SoundProcessingParams:
    """
    Параметры для алгоритма обработки аудиодорожек и характеристики самих записей.

    snr_db: набор значений signal-to-noise ratio
    sampling_rate: sr входной аудиодорожки
    pesq_sampling_rate: параметр fs для расчета Perceptual Evaluation Speech Quality
    pesq_mode: параметр mode для расчета Perceptual Evaluation Speech Quality
    """

    snr_db: List[int]
    sampling_rate: int
    pesq_sampling_rate: int
    pesq_mode: str


@dataclass
class Params:
    gdrive_params: GdriveParams
    sound_files: SoundFilePaths
    sound_proc_params: SoundProcessingParams
