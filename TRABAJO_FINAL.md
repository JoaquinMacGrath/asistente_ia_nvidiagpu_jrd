# Trabajo final: NVIDIA GPU & AI Assistant

**Entrega: jueves 5 de noviembre de 2026.**

## Qué haces

1. **Haz funcionar el asistente como bot de Telegram en Render**, usando la plantilla del curso.
2. **Especialízalo en GPU NVIDIA e IA.** El proyecto usa cinco whitepapers sobre Pascal, Turing, Ampere, Ada Lovelace y Blackwell como base documental.
3. **Amplíalo progresivamente** con diagnóstico de hardware, una API pública relacionada con IA, búsqueda web, un servidor MCP y un flujo con Pydantic Graph, explicando los supuestos y límites de sus respuestas.

## Qué entregas (por el canal que indique el docente)

- [ ] El **usuario de tu bot** (`@…_bot`). Antes, pídele al docente su número de chat y agrégalo a `TELEGRAM_PERMITIDOS` en Render.
- [ ] El **enlace a tu repositorio**. Puede ser privado si compartes el acceso con el docente. En su README describe el asistente y agrega preguntas para probarlo, por ejemplo una comparación entre arquitecturas y una consulta sobre capacidades de GPU.

## Cómo se califica

**Solo el funcionamiento.** 

1. responde en Telegram desde Render;
2. devuelve respuestas adecuadas
3. lo personalizaste

## Antes de entregar

- [ ] El bot responde **hoy** en Render. Si se durmió, el primer mensaje tarda alrededor de un minuto, y eso es normal.
- [ ] Un chat que no está en la lista recibe «No autorizado».
- [ ] `uv run pytest` pasa.
- [ ] `.env` no está en el repositorio y `git grep -nE "AIza[0-9A-Za-z_-]{30}"` no encuentra nada.

## Si la nube falla

Si Render, Telegram o la cuota del modelo te bloquean, avisa **antes del 5 de noviembre**. Entrega una grabación corta de la pantalla con el bot funcionando en tu equipo (`uv run bot.py`) y el error que te dio la nube.
