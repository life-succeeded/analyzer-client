from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from fastapi.responses import JSONResponse
import json

# Прямой импорт (без точки), так как запускаем из той же папки
from neuro import inspect_photo_quality

app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    # Проверяем, что файл является изображением
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    try:
        # Читаем содержимое файла в память
        contents = await file.read()

        # Преобразуем в numpy массив
        nparr = np.frombuffer(contents, np.uint8)

        # Декодируем изображение с помощью OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")

        # Получаем размеры изображения
        height, width, channels = img.shape

        # Конвертация в grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Анализ качества фото (теперь используем напрямую импортированную функцию)
        is_blured, blur_score, quality_score, has_error = inspect_photo_quality(img)

        # Возвращаем результат
        return JSONResponse({
            "is_blured": bool(is_blured),
            "blur_score": blur_score,
            "quality_score": quality_score,
            "has_error": bool(has_error),
            "filename": file.filename,
            "dimensions": f"{width}x{height}",
            "channels": channels,
            "message": "Изображение успешно обработано",
            "processed": "grayscale conversion done"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
