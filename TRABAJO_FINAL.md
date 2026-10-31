# VRAMind: descripción del proyecto

VRAMind es un asistente educativo para explorar arquitecturas y tecnologías de GPU NVIDIA, consultar información sobre modelos de inteligencia artificial y estimar qué modelos de generación de texto podrían probarse en un equipo local.

## Capacidades principales

- Consulta y comparación de Pascal, Turing, Ampere, Ada Lovelace y Blackwell mediante cinco documentos PDF de NVIDIA.
- Diagnóstico de hardware según GPU, VRAM, CPU y RAM, con estimaciones para diez modelos de texto y cuantizaciones Q4, Q5 y Q8.
- Búsqueda de modelos y consulta de metadatos públicos en Hugging Face Hub.
- Búsqueda web de información reciente mediante Tavily.
- Atención mediante bot de Telegram y API HTTP opcional.

Un grafo Pydantic Graph clasifica las consultas y las dirige al agente PydanticAI, que utiliza las herramientas existentes cuando corresponda.

## Limitaciones

Las estimaciones de memoria son aproximadas y no garantizan compatibilidad ni rendimiento. El offloading a RAM es una posibilidad teórica que depende del runtime y puede reducir considerablemente la velocidad.

El índice RAG se construye en memoria al realizar la primera consulta del proceso. Esa operación genera embeddings con Gemini y puede estar sujeta a sus límites de cuota.
