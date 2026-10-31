# API HTTP opcional de VRAMind

Además del bot de Telegram descrito en el [README](README.md), VRAMind puede atender solicitudes HTTP para que otros programas consulten el mismo agente.

La API es el archivo `api.py`. Recibe un mensaje por HTTP y devuelve la respuesta del **mismo** `responder(mensaje)` que usa el bot. Si cambias una tool en `capacidades.py`, la toman los dos.

```
POST /chat   Authorization: Bearer <tu token>   {"mensaje": "¿Qué diferencias hay entre Ampere y Ada Lovelace?"}
         →   {"texto": "…respuesta del asistente con citas a los documentos…"}
```

## 1 · Crea tu token

La API solo responde a quien mande un token configurado en `API_TOKENS`. Sin token válido, contesta `401` **sin llamar al modelo**. Si la lista está vacía, no entra nadie.

1. Genera un token:
   ```bash
   uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. Ponlo en `.env`:
   ```
   API_TOKENS=el-token-que-salió
   ```
   Para varias personas, genera un token para cada una y sepáralos con comas. Si uno se filtra, borras solo ese.

> **No reutilices el token de Telegram.** Ese token es la llave del bot: quien lo tenga puede leer sus mensajes y hacerse pasar por él. Cada cosa lleva su propia llave.

## 2 · Pruébala en tu equipo

```bash
uv run uvicorn api:app --reload
```

- Abre **http://127.0.0.1:8000/docs**: FastAPI arma una página para probarla. Pulsa **Authorize**, pega tu token y luego, en `POST /chat`, *Try it out*.
- O desde la terminal:
  ```bash
  curl -X POST http://127.0.0.1:8000/chat \
    -H "Authorization: Bearer el-token-que-salió" \
    -H "Content-Type: application/json" \
  -d '{"mensaje": "¿Qué diferencias hay entre Ampere y Ada Lovelace?"}'
  ```

| Respuesta | Qué significa |
|---|---|
| `200` y `{"texto": …}` | Funciona |
| `401 Token inválido` | Falta el token, está mal o `API_TOKENS` está vacío |
| `422` | El pedido no cumple el contrato: falta `mensaje`, está vacío o pasa de 2000 caracteres |
| `500` | Falló el agente o una tool. El cliente recibe un mensaje genérico; el registro del servidor contiene solo el tipo de excepción y un identificador de incidente, no el texto interno del error. Un límite del proveedor puede aparecer como respuesta de la tool dentro de `200`. |

## 3 · Publícala en Render

Es un **segundo servicio** del mismo repositorio. El servicio del bot no se toca.

1. En [dashboard.render.com](https://dashboard.render.com): **+ New → Web Service** y elige el mismo repositorio.
2. Completa los campos como los del bot, salvo el nombre y el **Start Command**:

| Campo | Valor |
|---|---|
| **Name** | `vramind-api` (o un nombre disponible). Será parte de la URL |
| **Build Command** | `uv sync --locked --no-dev`, el mismo |
| **Start Command** | `uv run --no-sync uvicorn api:app --host 0.0.0.0 --port $PORT` |
| **Instance Type** | **Free** |

3. En **Environment Variables**:

| Key | Value |
|---|---|
| `GEMINI_API_KEY_LOCAL` | Tu clave de Gemini |
| `API_TOKENS` | Tu token o tus tokens |

4. Pulsa **Deploy Web Service** y pruébala como en el paso 2, usando la URL que Render asigne en lugar de `http://127.0.0.1:8000`.

Cada `git push` vuelve a desplegar los dos servicios. Los límites del plan Free son los mismos que los del bot: se duerme a los 15 minutos, el disco es efímero y las 750 horas gratis al mes se reparten entre tus servicios (alcanzan, porque se duermen).

## Cuidado

- El token sirve cuando la API la llama **un programa**. Si haces una página web que llama a la API **desde el navegador**, cualquiera puede ver el token en las herramientas de desarrollo. Para eso hace falta un login o un servidor intermedio.
- Nunca escribas un token en el código ni lo subas a GitHub: va en `.env` y en *Environment*.
- Las respuestas HTTP ante errores del agente son genéricas. El registro guarda el tipo de excepción y un identificador aleatorio de incidente, sin incluir el contenido de la excepción.
- `uv run pytest` ejecuta la suite automatizada; las pruebas de autenticación verifican que un token inválido no llama al agente y que uno válido permite procesar la solicitud con un modelo simulado.
