# CameraVBG  - Camera Video Background Removal

Este proyecto implementa una solución avanzada de segmentación de personas y sustitución de fondo en tiempo real. Utiliza el motor de **MediaPipe** para la inferencia y una serie de filtros de post-procesamiento para lograr un acabado profesional, minimizando los artefactos visuales y el parpadeo.

## ✨ Características Principales

* **Inferencia Optimizada:** Basado en MediaPipe (TFLite), diseñado para ejecutarse con baja latencia.
* **Estabilidad Temporal:** Filtro de suavizado exponencial para eliminar el "ruido" o parpadeo en los bordes de la máscara.
* **Armonización de Color (Color Transfer):** Ajusta automáticamente el tono del sujeto para que coincida con la iluminación del fondo mediante el espacio de color LAB.
* **Light Wrap:** Técnica de composición que permite que la luz del fondo "envuelva" los bordes del sujeto, mejorando el realismo de la integración.
* **Refinado de Bordes:** Soporte para **Guided Filter** y operaciones morfológicas para capturar detalles finos y limpiar la máscara.
* **Cámara Virtual:** Integración con `pyvirtualcam` para enviar la salida directamente a aplicaciones como Zoom, Teams, OBS o Discord.
* **Captura Asíncrona:** Hilo de lectura de cámara independiente para maximizar los FPS.


Se recomienda usar un entorno virtual de Python 3.12+, con el gestor de paquetes `uv`.