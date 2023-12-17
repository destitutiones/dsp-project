import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
import soundfile as sf

from dsp_project.utils import calculate_metrics, prepare_data

from df.enhance import enhance, init_df, load_audio, save_audio  # isort: skip


dsp_project_dir_path = Path(__file__).absolute().parent.parent
root_dir_path = dsp_project_dir_path.parent
sys.path.append(str(dsp_project_dir_path))
sys.path.append(str(root_dir_path.joinpath("configs")))

from configs.hw3.config import Params  # noqa: E402


@hydra.main(config_path="../configs/hw3", config_name="config", version_base="1.3")
def main(cfg: Params):
    # Все пути, константы и параметры запуска берем из конфига
    # Скачивание аудиодорожек из гугл-диска
    if_download_files = cfg["gdrive_params"]["if_download_files"]
    gdrive_sound_zip_path = cfg["gdrive_params"]["gdrive_sound_zip_path"]
    # Пути до данных
    data_dir_path = cfg["sound_files"]["data_dir_path"]
    zip_file_name = cfg["sound_files"]["zip_file_name"]
    speech_file_name = cfg["sound_files"]["speech_file_name"]
    # Параметры для расчета метрик
    snr_db_vals = cfg["sound_proc_params"]["snr_db"]
    sampling_rate = cfg["sound_proc_params"]["sampling_rate"]
    pesq_sampling_rate = cfg["sound_proc_params"]["pesq_sampling_rate"]
    pesq_mode = cfg["sound_proc_params"]["pesq_mode"]

    if if_download_files:
        prepare_data(
            gdrive_sound_zip_path, data_dir_path + zip_file_name, data_dir_path
        )

    logging.info("Loading input files")
    speech, _ = sf.read(data_dir_path + speech_file_name)

    metrics_list = []

    logging.info("Processing mixed files")
    for snr_db in [None] + snr_db_vals:
        mix_file_name = speech_file_name if snr_db is None else f"mixed_{snr_db}.wav"

        logging.info(f"Enhancing speech and noise with snr={snr_db}dB")

        model, df_state, _ = init_df()  # Load default model
        sample_rate = df_state.sr()

        noisy_audio, _ = load_audio(data_dir_path + mix_file_name, sr=sample_rate)
        enhanced_audio = enhance(model, df_state, noisy_audio)

        mix_file_df2_name = mix_file_name.split(".")
        mix_file_df2_name = f"{mix_file_df2_name[0]}_df2enhanced.{mix_file_df2_name[1]}"

        save_audio(data_dir_path + mix_file_df2_name, enhanced_audio, sr=sample_rate)

        metrics = calculate_metrics(
            speech, enhanced_audio, sampling_rate, pesq_sampling_rate, pesq_mode
        )
        metrics["file"] = mix_file_df2_name
        metrics["SNR, dB"] = snr_db
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    print(metrics_df[["file", "SNR, dB", "SDR", "SI-SDR", "PESQ"]])


if __name__ == "__main__":
    main()
