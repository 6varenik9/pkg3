# Image Processing App

## Описание
**Image Processing App** — это приложение для обработки изображений, позволяющее пользователям загружать изображения и применять различные методы обработки, такие как пороговая обработка, детекция краев, линий и точек. Программа предоставляет удобный интерфейс для визуализации оригинальных и обработанных изображений.

## Установка и запуск

### Системные требования
- **Операционная система**: Windows, macOS, или Linux.
- **Python**: 3.x (необходим для запуска из исходного кода).
- **Библиотеки**: PyQt5, OpenCV, Pillow, NumPy.

### Запуск приложения
1. **Использование исполняемого файла**:
   - Скачайте файл `ImageProcessingApp.exe` (если доступно).
   - Дважды щелкните по файлу для запуска приложения.

2. **Запуск из исходного кода**:
   - Убедитесь, что установлен Python версии 3.x.
   - Скачайте исходный код приложения из репозитория GitHub.
   - Установите необходимые библиотеки:
     ```bash
     pip install PyQt5 opencv-python Pillow numpy
     ```
   - Запустите приложение:
     ```bash
     python ImageProcessingApp.py
     ```

## Пользовательский интерфейс

### Основные элементы
- **Кнопка "Load Image"**: открывает диалог для выбора изображения.
- **Комбобокс**: позволяет выбрать метод обработки изображения.
- **Спин-бокс**: для установки порога обработки (если применимо).
- **Кнопка "Apply"**: применяет выбранный метод обработки к загруженному изображению.
- **Фреймы**: отображают оригинальное и обработанное изображения.

### Навигация по интерфейсу
- **Загрузка изображения**:
  - Нажмите кнопку "Load Image".
  - Выберите изображение в открывшемся диалоговом окне.
  - Загруженное изображение будет отображено слева, а обработанное — справа (после применения метода).

- **Применение методов обработки**:
  - Выберите желаемый метод из комбобокса.
  - Установите порог (если требуется).
  - Нажмите кнопку "Apply" для применения выбранного метода.

## Особенности приложения

- **Поддержка различных методов обработки**: Программа предлагает несколько методов, включая глобальную и адаптивную пороговую обработку, детекцию краев, линий и точек.

- **Гибкость в выборе параметров**: Пользователь может настраивать параметры обработки, такие как порог для методов пороговой обработки.

## Известные ограничения
- **Поддерживаемые форматы**: Программа работает только с изображениями в форматах PNG, JPG и BMP.
- **Работа с большими изображениями**: Обработка больших изображений может занимать больше времени и ресурсов.

## Исходный код
Исходный код приложения доступен на GitHub. Вы можете ознакомиться с кодом и внести свои предложения или доработки.
