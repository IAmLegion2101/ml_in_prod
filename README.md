# HW 1 Heart disease classification

## Разработать проект для классификации болзней сердца

## Данные
Использован следующий источник данных:

    https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

## Подготовка рабочей среды

Необходимо настроить виртуальную среду. Для этого нужно выполнить следующие команды:
1) в консоли перейти в папку с проектом;
2) создать виртуальную среду, выполнив команду python -m venv heart_dis_env;
3) активировать среду командой heart_dis_env\Scripts\activate;
4) установить все необходимые модули, воспользовавшись командой pip install -r requirements.txt

Необходимо настроить структуру проекта, выполнив bat-файл

    prepare_for_work_env.bat

В результате будут созданы папки:
* logs
* latest_trained
* predictions
* reports

## Обучение модели

Обучение модели производится командой:

    python src\train.py

По завершиению обучения в папке reports появится подпапка с датой
и временем завершения обучения, в которой будут находиться следующие файлы:
* general_config.yaml - конфиг с основными параметрами;
* model.pickle - дамп обученной модели
* score_report.yaml - отчет по оценкам модели
* также в папке latest_trained будет создана копия дампа модели для использования в прогнозе.

## Прогноз

Запуск прогноза выполняется следующей командой:

    python src\predict.py

Скрипт принимает на вход следующие аргументы:
* -h, --help - вызов справочной информации;
* -mp, --model_path - путь до дампа модели;
* -r, --res_path - путь до места хранения результата;
* -t, --target - колонка с предсказываемыми значениями (опционально)

## Структура проекта:
    D:.
    │   .gitignore
    │   img.png
    │   prepare_work_env.bat
    │   README.md
    │   requirements.txt
    │
    ├───conf
    │   │   common_conf.yaml
    │   │   logging.conf
    │   │
    │   ├───data
    │   │       data_conf.yaml
    │   │
    │   ├───data_work
    │   │       data_work_params.yaml
    │   │
    │   ├───general
    │   │       general_conf.yaml
    │   │
    │   └───model
    │           gaussiannb.yaml
    │           random_forest_cls.yaml
    │           xgboost.yaml
    │
    ├───data
    │       heart_cleveland_upload.csv
    │
    ├───latest_trained
    ├───logs
    ├───notebooks
    │       hw_1_baseline.ipynb
    │
    ├───predictions
    ├───reports
    └───src
        │   logging.conf
        │   predict.py
        │   prepare_to_run.py
        │   train.py
        │
        ├───config
        │       conf.py
        │       data_conf.py
        │       general_conf.py
        │       model_conf.py
        │       __init__.py
        │
        ├───data
        │       prepare_data.py
        │       __init__.py
        │
        └───utils
                utils.py





