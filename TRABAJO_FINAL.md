# Trabajo final: VRAMind

**Entrega: jueves 5 de noviembre de 2026.**

## Qué haces

1. **Haz funcionar el asistente como bot de Telegram en Render**, usando la plantilla del curso.
2. **Especialízalo en GPU NVIDIA e IA.** El proyecto usa cinco whitepapers sobre Pascal, Turing, Ampere, Ada Lovelace y Blackwell como base documental.
3. **Amplíalo progresivamente** con diagnóstico de hardware, una API pública relacionada con IA, búsqueda web y un flujo con Pydantic Graph, explicando los supuestos y límites de sus respuestas.

## Estado implementado

VRAMind conserva un agente PydanticAI con cuatro capabilities: RAG sobre cinco whitepapers PDF de NVIDIA (Pascal, Turing, Ampere, Ada Lovelace y Blackwell), diagnóstico de hardware para diez modelos de texto y cuantizaciones Q4/Q5/Q8, consultas públicas de Hugging Face Hub y búsqueda actualizada mediante Tavily. Pydantic Graph clasifica cada consulta localmente y conecta rutas específicas, combinadas o generales con el agente existente. El bot de Telegram mantiene restricciones por ID de chat y la API FastAPI opcional exige un token configurado.

El diagnóstico es una estimación aproximada: no garantiza ejecución, compatibilidad ni velocidad. El offloading a RAM es una posibilidad teórica dependiente del runtime. El índice RAG es efímero y se crea al iniciar el proceso en la primera consulta, consumiendo embeddings de Gemini.

La integración de servidor MCP mencionada en versiones anteriores de la planificación no forma parte de VRAMind y no está implementada. Si MCP sigue siendo un requisito obligatorio de la rúbrica vigente del docente, queda pendiente confirmar si esta exclusión es aceptable.

## Qué entregas (por el canal que indique el docente)

- [ ] El **usuario de tu bot** (`@…_bot`). Antes, pide al docente el ID de su chat privado y agrégalo a `TELEGRAM_PERMITIDOS` en Render.
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
