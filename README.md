# dsp-project

MIPT DSP course project, autumn 2023

# Запуск проекта

## Пререквизиты

[Установка poetry](https://python-poetry.org/docs/#installation)

## Запуск

```bash
python3 -m venv dsp_env
source dsp_env/bin/activate
poetry install
pre-commit install
pre-commit run -a
python3 dsp_project/hw1_deconv.py
```
