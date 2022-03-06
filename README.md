## Распознавание рукописного текста в школьных тетрадях
Соревнование, проводимое в рамках олимпиады НТО, разработанное Сбером. Платформа [ODS](https://ods.ai/competitions/nto_final_21-22).

### Задача
> Вам нужно разработать алгоритм, который способен распознать рукописный текст в школьных тетрадях. В качестве входных данных вам будут предоставлены фотографии целых листов. Предсказание модели — список распознанных строк с координатами полигонов и получившимся текстом.

### Как должно работать решение?
> Последовательность двух моделей: сегментации и распознавания. Сначала сегментационная модель предсказывает полигоны маски каждого слова на фото. Затем эти слова вырезаются из изображения по контуру маски (получаются кропы на каждое слово) и подаются в модель распознавания. В итоге получается список распознанных слов с их координатами.

### Ресурсы
**Christofari** с **NVIDIA Tesla V100** и образом **jupyter-cuda10.1-tf2.3.0-pt1.6.0-gpu:0.0.82**

### Обучение

**Instance Segmentation** - модель [X101-FPN](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-instance-segmentation-baselines-with-mask-r-cnn) из зоопарка моделей detectron2 + аугментации + высокое разрешение\
**OCR** - архитектура CRNN с бекбоном Resnet-34, предобученным на топ 1 модели соревнования [Digital Peter](https://github.com/sberbank-ai/digital_peter_aij2020) + аугментации\
**Beam Search** - модель KenLM, обученная на данных сорвенования Feedback + задачи с тектом из Решу ОГЭ/ЕГЭ, а также CTCDecoder 
