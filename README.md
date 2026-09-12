# Lumen RunPod Worker

Worker Serverless para transformar video con Wan VACE y devolver MP4/H.264 con el audio original.

Entrada: `video_url`, `prompt`, `style` (`realistic`, `comic`, `manga`, `hybrid`), `preserve_audio` y `mode` (`clip` o `extended`).

## Despliegue

- Prueba: `Wan-AI/Wan2.1-VACE-1.3B-diffusers`, GPU de 24 GB.
- Alta calidad: `Wan-AI/Wan2.1-VACE-14B-diffusers`, GPU de 80 GB y `LUMEN_MODEL_ID` con ese identificador.
- Montar Network Volume en `/runpod-volume` para conservar los pesos.
- Guardar la API key solo como secreto del backend de Lumen, nunca en el navegador.

Normaliza a 832x480, 16 fps, H.264 Main/yuv420p y AAC 48 kHz; procesa segmentos de 81 fotogramas, recompone la duración y reincorpora el audio original.
