## Распознавание рукописного текста в школьных тетрадях
#### Соревнование, проводимое в рамках олимпиады НТО, разработанное Сбером. Платформа [ODS](https://ods.ai/competitions/nto_final_21-22).

### [Результаты Public](https://ods.ai/competitions/nto_final_21-22/leaderboard)
![leaderbord](https://github.com/Lednik7/nto-ai-text-recognition/raw/main/images/public_leaderbord.jpg)

### Задача
> Вам нужно разработать алгоритм, который способен распознать рукописный текст в школьных тетрадях. В качестве входных данных вам будут предоставлены фотографии целых листов. Предсказание модели — список распознанных строк с координатами полигонов и получившимся текстом.
---

### Как должно работать решение?
> Последовательность двух моделей: сегментации и распознавания. Сначала сегментационная модель предсказывает полигоны маски каждого слова на фото. Затем эти слова вырезаются из изображения по контуру маски (получаются кропы на каждое слово) и подаются в модель распознавания. В итоге получается список распознанных слов с их координатами.
---

### Модели

**[Instance Segmentation](https://colab.research.google.com/github/Lednik7/nto-ai-text-recognition/blob/main/train/detectron2_segmentation_latest.ipynb)**
- модель [X101-FPN](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn) из зоопарка моделей detectron2 + аугментации + высокое разрешение

**[Optical Character Recognition (OCR)](https://colab.research.google.com/github/Lednik7/nto-ai-text-recognition/blob/main/train/ocr_model.ipynb)**
- архитектура CRNN с бекбоном Resnet-34, предобученным на топ 1 модели соревнования [Digital Peter](https://github.com/sberbank-ai/digital_peter_aij2020) + аугментации

**[Beam Search](https://colab.research.google.com/github/Lednik7/nto-ai-text-recognition/blob/main/dataset/make_kenlm_dataset_latest.ipynb)**
- модель KenLM, обученная на данных сорвенования [Feedback](https://www.kaggle.com/c/feedback-prize-2021/data ) + задачи с тектом из Решу ОГЭ/ЕГЭ, а также [CTCDecoder](https://github.com/parlance/ctcdecode)

### Ресурсы & Submit
---
**Christofari** с **NVIDIA Tesla V100** и образом **jupyter-cuda10.1-tf2.3.0-pt1.6.0-gpu:0.0.82**

Мы не гарантируем поддержку сабмита всё время, поэтому предоставляем 2 ссылки:
[Google Drive](https://drive.google.com/file/d/13jbbnSuwn5g4ml_DIcvDm7AI1dMS8j4L/view?usp=sharing) и 
[Yandex Drive](https://storage.yandexcloud.net/datasouls-ods/submissions/e7c3d807-0f20-4003-9935-977432b4d615/14eafde9/sub_8%281%29.zip)
