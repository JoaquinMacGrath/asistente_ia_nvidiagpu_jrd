# VRAMind

VRAMind es un asistente educativo para consultar arquitecturas y tecnologías NVIDIA y estimar qué modelos generativos de texto podrían probarse localmente en un equipo. Está construido con Python, PydanticAI, Pydantic Graph, ChromaDB, Hugging Face Hub, Tavily, Telegram y FastAPI.

## Capacidades

- **Documentación NVIDIA (RAG):** cinco whitepapers PDF sobre Pascal, Turing, Ampere, Ada Lovelace y Blackwell, con generación, chip, tema, fuente, documento y página en los metadatos. `datos/documentos/fuentes_web.md` es una lista manual de referencias y no se indexa en el RAG.
- **Diagnóstico local:** diez modelos de texto de Qwen, Llama, Gemma, Phi y DeepSeek; estima automáticamente Q4, Q5 y Q8 y separa pesos, caché KV y runtime.
- **Hugging Face Hub:** búsqueda pública y consulta de metadatos/configuración de modelos. No modifica el catálogo local ni descarga pesos.
- **Tavily:** búsqueda web actualizada con títulos, fragmentos y enlaces de hasta cinco resultados.
- **Pydantic Graph:** clasifica la consulta con reglas locales, dirige el turno a una ruta y construye la respuesta usando el agente PydanticAI existente. Las rutas son NVIDIA, hardware, Hugging Face, web, combinada y general.
- **Telegram:** polling local o webhook en Render. En chats privados conserva el historial en memoria hasta reiniciar el proceso; `/start` lo reinicia.
- **API opcional:** FastAPI, protegida con `API_TOKENS`; ver [API.md](API.md).

El agente responde en español. No hay integración MCP.

## Diagnóstico: alcance y límites

Indica GPU, VRAM, CPU y RAM; el contexto predeterminado es 4096 tokens. Se aplican reservas de VRAM y RAM. Se muestran modelos que caben en VRAM estimada, opciones con posible offloading y modelos que superan la memoria estimada. Se excluyen y explican por separado modelos cuyo límite de contexto sea inferior al solicitado.

Q4, Q5 y Q8 son supuestos aproximados de bytes por parámetro, no tamaños de archivos reales. El offloading es una posibilidad teórica basada en memoria: depende del runtime, puede reducir mucho la velocidad y no garantiza que el modelo pueda ejecutarse. Las estimaciones no verifican drivers, compatibilidad o rendimiento real.

## Requisitos y configuración local

Se necesita Python 3.14 y [`uv`](https://docs.astral.sh/uv/). Copia `.env.example` como `.env` y completa solo lo necesario. Nunca subas `.env` ni pegues claves en el código.

```dotenv
MODO=gemini
GEMINI_API_KEY_LOCAL=tu_clave
GEMINI_MODEL=gemini-3.5-flash-lite
OLLAMA_MODEL=gemma4:e2b
TELEGRAM_TOKEN=token_de_botfather
TELEGRAM_PERMITIDOS=id_de_chat_privado
TAVILY_API_KEY=tu_clave_tavily
API_TOKENS=
```

`GEMINI_API_KEY_LOCAL` es necesaria si `MODO=gemini`; en `MODO=local` se usa Ollama en `http://localhost:11434/v1`. Hugging Face consulta recursos públicos sin token. Tavily requiere `TAVILY_API_KEY` para búsquedas web. `API_TOKENS` es opcional y solo se necesita para habilitar clientes de la API.

Para Telegram, `TELEGRAM_PERMITIDOS` debe contener el ID del chat privado autorizado (IDs separados por coma si hay más usuarios). No hace falta habilitar grupos. El bot rechaza chats no autorizados antes de llamar al agente.

Instalación y ejecución:

```bash
uv sync
uv run asistente.py
uv run bot.py
```

La API opcional se inicia con:

```bash
uv run uvicorn api:app --reload
```

Las pruebas automatizadas usan modelos, herramientas y respuestas simulados; no requieren consumir cuotas de Gemini, Tavily o Hugging Face:

```bash
uv run pytest
```

## RAG y cuotas de embeddings

El índice utiliza una colección ChromaDB efímera y se reconstruye al iniciar cada proceso cuando se realiza la primera búsqueda RAG. La creación del índice extrae los PDF y solicita embeddings a Gemini en lotes, por lo que esa primera consulta puede consumir cuota de embeddings. Las búsquedas posteriores reutilizan el índice mientras ese proceso siga activo.

## Render

`render.yaml` despliega el bot como servicio web, instala con `uv sync --locked --no-dev` y ejecuta `uv run --no-sync bot.py`. Configura en Render **Environment** las variables `GEMINI_API_KEY_LOCAL`, `TELEGRAM_TOKEN`, `TELEGRAM_PERMITIDOS` y `TAVILY_API_KEY`. No guardes valores reales en `render.yaml`. Render define `RENDER_EXTERNAL_URL` y `PORT` para activar el webhook. Para Ollama usa el modo local solo en un equipo donde ese servicio esté disponible; no está desplegado en Render.

El almacenamiento de Render es efímero: el historial de Telegram y el índice RAG en memoria no sobreviven a un reinicio. No ejecutes polling local al mismo tiempo que el webhook con el mismo token.

## Archivos principales

- [asistente.py](asistente.py): agente único y puntos de entrada `responder()` y `responder_con_conversacion()`.
- [grafo.py](grafo.py): clasificación local y flujo condicional Pydantic Graph.
- [capacidades.py](capacidades.py): RAG, diagnóstico, Hugging Face y Tavily.
- [config.py](config.py): ajustes y secretos con `SecretStr`.
- [bot.py](bot.py): Telegram y control de acceso.
- [api.py](api.py): API opcional.
- [datos/documentos/](datos/documentos/): cinco PDF y referencias web manuales.
- [tests/](tests/): pruebas locales automatizadas.

Más detalles del proyecto: [descripción de VRAMind](TRABAJO_FINAL.md) y [documentación de la API](API.md).
