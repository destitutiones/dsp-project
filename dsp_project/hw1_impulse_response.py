import logging
import sys
from pathlib import Path

import hydra
import soundfile as sf

from dsp_project.hw1_sound_processing import process_sound
from dsp_project.utils import plot_input_tracks, prepare_data

dsp_project_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = dsp_project_dir_path.parent
sys.path.append(str(dsp_project_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.hw1.config import Params  # noqa: E402


@hydra.main(config_path="../configs/hw1", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    # Все пути, константы и параметры запуска берем из конфига
    # Скачивание аудиодорожек из гугл-диска
    if_download_files = cfg["gdrive_params"]["if_download_files"]
    gdrive_sound_zip_path = cfg["gdrive_params"]["gdrive_sound_zip_path"]
    # Пути до данных
    data_dir_path = cfg["sound_files"]["data_dir_path"]
    zip_file_name = cfg["sound_files"]["zip_file_name"]
    sweeper_original_file_name = cfg["sound_files"]["sweeper_original_file_name"]
    sweeper_record_file_name = cfg["sound_files"]["sweeper_record_file_name"]

    if if_download_files:
        prepare_data(
            gdrive_sound_zip_path, data_dir_path + zip_file_name, data_dir_path
        )

    logging.info("Loading sweeper files")
    orig, sr = sf.read(data_dir_path + sweeper_original_file_name)
    reverb, _ = sf.read(data_dir_path + sweeper_record_file_name)

    # Предварительно обрезали запись с диктофона, поэтому размеры файлов должны быть одинаковыми
    assert (
        orig.shape == reverb.shape
    ), "Original and recorded audio tracks should have equal shapes"

    ## файлы должны быть не только обрезаны, а еще и выравнены по началу; это можно было сделать через кросс-корреляцию
    ## или через добавление маркера в аудио (искусственно забиваешь единицами примерно 10мс в самом начале, и в записи обрезаешь по этому маркеру)

    plot_input_tracks(
        {"original": orig, "record": reverb},
        "Plotting input audio tracks amplitude graph",
        "Input audio tracks",
        "Time",
        "Amplitude",
    )

    logging.info("Processing sweeper files")

    process_sound(cfg["sound_proc_params"], cfg["sound_files"], orig, reverb)
    logging.info("Finished")


if __name__ == "__main__":
    main()
