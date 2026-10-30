# NVIDIA GPU & AI Assistant

Asistente educativo especializado en arquitecturas y tecnologías NVIDIA para inteligencia artificial. El proyecto reutiliza la plantilla del curso «Más allá de los chatbots», con Pydantic AI, Gemini, ChromaDB, Telegram y Render.

## Estado actual

- Un agente principal responde en español mediante `responder(mensaje)`.
- El RAG indexa los PDF de `datos/documentos/` al realizar la primera consulta y devuelve fragmentos con identificadores de fuente.
- Telegram ofrece polling local y webhook en Render. Solo responde a los chats incluidos en `TELEGRAM_PERMITIDOS`.
- En chats privados, Telegram conserva el historial de la conversación en memoria durante la ejecución; `/start` lo reinicia y un reinicio del proceso lo pierde. En grupos, cada mensaje se procesa sin historial compartido.
- La API HTTP de `api.py` es opcional y usa el mismo agente; requiere `API_TOKENS`.

El asistente se centra en Pascal, Turing, Ampere, Ada Lovelace y Blackwell. Las instrucciones le piden distinguir las capacidades de una arquitectura de las de un modelo concreto de GPU y citar los fragmentos recuperados.

La colección RAG guarda metadatos de generación, arquitectura documentada, tema, fuente y PDF. En comparaciones, la búsqueda recupera resultados por separado para cada generación solicitada. La colección es temporal y se reconstruye cuando inicia el proceso.

El diagnóstico local de modelos de texto usa un catálogo ampliable de cinco tamaños Qwen3. Separa memoria estimada de pesos, caché KV y runtime; reserva parte de la VRAM y RAM para el entorno, y señala cuándo podría hacer falta offloading. Sus factores de cuantización y runtime son supuestos, no tamaños exactos de archivos. La búsqueda web general, otras APIs, MCP y Pydantic Graph están previstos para etapas posteriores.

La capability de Hugging Face Hub busca hasta cinco modelos públicos y consulta metadatos, configuración y un fragmento limitado de su tarjeta. No descarga pesos ni modifica el catálogo Qwen3; el contenido de las tarjetas se trata como fuente externa no confiable.

## Requisitos y configuración

Se necesita `uv`, una clave de Gemini y, para usar Telegram, un bot creado con [@BotFather](https://telegram.org/). Copia `.env.example` como `.env` y completa las variables necesarias:

```dotenv
MODO=gemini
GEMINI_API_KEY_LOCAL=
GEMINI_MODEL=gemini-3.5-flash-lite
TELEGRAM_TOKEN=
TELEGRAM_PERMITIDOS=
API_TOKENS=
```

No subas `.env` ni incluyas claves en el código. En Render, configura las variables en **Environment**.

## Ejecutar localmente

```bash
uv sync
uv run asistente.py
uv run bot.py
```

Para probar la API opcional:

```bash
uv run uvicorn api:app --reload
```

El endpoint `POST /chat` requiere `Authorization: Bearer <token>` y acepta un JSON con `mensaje`.

## Despliegue en Render

El archivo `render.yaml` configura un servicio web con `uv sync --locked --no-dev` y `uv run --no-sync bot.py`. Añade `GEMINI_API_KEY_LOCAL`, `TELEGRAM_TOKEN` y `TELEGRAM_PERMITIDOS` en las variables de entorno del servicio.

Render activa el webhook automáticamente mediante `RENDER_EXTERNAL_URL`. En local, el bot usa polling. No ejecutes simultáneamente dos instancias en polling con el mismo token.

## Archivos principales

- `asistente.py`: instrucciones del agente y función `responder`.
- `capacidades.py`: RAG NVIDIA, estimador de memoria y consulta de modelos en Hugging Face Hub.
- `config.py`: configuración del modelo y secretos.
- `bot.py`: bot de Telegram y control de acceso.
- `api.py`: API HTTP opcional.
- `datos/documentos/`: cinco whitepapers de arquitecturas NVIDIA y `fuentes_web.md`.
- `tests/`: pruebas de acceso a Telegram y autenticación de la API con modelos falsos.

La guía del curso está en `.agents/skills/asistente-pydantic-ai/` y los requisitos de entrega en `TRABAJO_FINAL.md`.
