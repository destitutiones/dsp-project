from dataclasses import dataclass


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
    sweeper_original_file_name: название файла с оригинальным свипером
    sweeper_record_file_name: название файла с записью свипера на микрофон
    noise_white_file_name: название файла с белым шумом
    noise_white_record_file_name: название файла с записью белого шума
    noise_white_adjusted_file_name: название файла со скорректированным белым шумом
    noise_pink_file_name: название файла с розовым шумом
    test_file_name: название оригинала тестового файла
    test_file_record_name: название записи тестового файла на диктофон
    test_file_record_conv_name: название записи тестового файла на диктофон с применением отклика
    """

    data_dir_path: str
    zip_file_name: str
    sweeper_original_file_name: str
    sweeper_record_file_name: str
    noise_white_file_name: str
    noise_white_record_file_name: str
    noise_white_adjusted_file_name: str
    noise_pink_file_name: str
    test_file_name: str
    test_file_record_name: str
    test_file_record_conv_name: str


@dataclass
class SoundProcessingParams:
    """
    Параметры для алгоритма обработки аудиодорожек и характеристики самих записей.

    band_cnt: размер "полосы" для разбивки частотной области
    """

    band_cnt: int
    max_sweeper_freq: int
    min_sweeper_freq: int
    sampling_rate: int


@dataclass
class Params:
    gdrive_params: GdriveParams
    sound_files: SoundFilePaths
    sound_proc_params: SoundProcessingParams
