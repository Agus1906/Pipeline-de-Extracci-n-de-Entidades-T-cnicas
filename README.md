# Pre-entrega 2 · Pipeline de Extracción de Entidades Técnicas

Pipeline asíncrono en Python 3.12+: texto sin procesar → prompt modular → LLM →
objeto Pydantic validado. Compatible con OpenAI y Anthropic mediante LangChain.

## Preparación y ejecución

1. Crear un entorno e instalar dependencias desde la raíz del repositorio:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
   Copy-Item .env.example .env
   ```

2. Editar `.env`: elegir `LLM_PROVIDER=openai` o `anthropic`, completar solamente
   la clave y el modelo de ese proveedor. El identificador debe estar habilitado
   en la cuenta y soportar tool calling. No se incluye ninguna clave real.
   `LLM_MAX_TOKENS=1024` y `LLM_TIMEOUT=30` son configurables; el timeout HTTP
   no es un plazo global de toda la cadena. El entorno del sistema tiene prioridad.
   `.env` se busca junto a `chain.py`.

3. Ejecutar el mini-script asíncrono:

   ```powershell
   .\.venv\Scripts\python.exe demo_async.py
   .\.venv\Scripts\python.exe demo_async.py --stress
   ```

   En Linux/macOS utilizar `python3.12` y `.venv/bin/python`.
   La demo requiere conexión, credenciales y cuota: cada caso puede generar hasta
   tres solicitudes facturables. El caso ambiguo sin tecnologías de `--stress`
   debe terminar rechazado si el modelo sigue las instrucciones; no se inventan
   tecnologías para satisfacer el esquema. Retorna 1 si algún caso falla y 130
   si se cancela con Ctrl+C. No reintentar indefinidamente.

## Archivos y contrato

| Archivo | Responsabilidad |
| --- | --- |
| `schemas.py` | Pydantic: tecnologías no vacías y sin duplicados, enum de criticidad, resumen acotado y campos extra prohibidos |
| `chain.py` | Configuración, ChatPromptTemplate, LCEL, validación de metadata, retry y `process_text` |
| `demo_async.py` | Entrada de ejemplo y prueba de estrés con `asyncio.run` |
| `test_pipeline.py` | Pruebas sin red ni claves reales |

`process_text(text: str)` devuelve `EntidadesTecnicas`, no un string JSON.
Rechaza entradas vacías o mayores a 20000 caracteres antes de crear el cliente.

```python
import asyncio
from chain import process_text

async def main():
    result = await process_text(
        "FastAPI usa Redis y PostgreSQL. La API está caída por conexiones agotadas."
    )
    print(result.model_dump_json(indent=2))

asyncio.run(main())
```

Salida ilustrativa; la redacción depende del modelo:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API FastAPI caída por agotamiento de conexiones a PostgreSQL; también utiliza Redis."
}
```

La criticidad es un **criterio propuesto para la actividad**, no una regla de
negocio validada: alta para caída/pérdida de datos/indisponibilidad; media para
degradación o impacto incierto; baja para descripciones sin incidentes. El prompt
solicita explicitar ambigüedades. La validación estructural no demuestra que una
extracción sea semánticamente correcta: evaluar ejemplos reales antes de usarla
en producción. No pegar logs con credenciales o datos sensibles.

## Cadena, parser y resiliencia

La composición central es:

```python
structured = model.with_structured_output(
    EntidadesTecnicas, method="function_calling", include_raw=True
)
chain = prompt | structured | RunnableLambda(validate_output)
```

`with_structured_output` integra el parser Pydantic. Se usa `function_calling`
en ambos proveedores para conservar metadata cruda; `include_raw=True` permite
detectar `finish_reason`/`stop_reason` antes de consumir el objeto parseado.
LangChain puede intentar parsearlo internamente; esta aplicación no lo acepta
sin revisar la finalización. `length`, `max_tokens`, metadata ausente, bloqueo,
JSON mal formado, campos faltantes y validación fallida nunca devuelven éxito.

Un runnable de observación envuelve la cadena con `.with_retry()`: máximo tres
intentos totales (dos reintentos), espera exponencial con jitter y ejecución
mediante `.ainvoke()`. Se repite la extracción con el mismo texto; no se reparan
campos a mano ni se promete recuperación de un texto sin evidencia. Después del
último intento, la excepción se propaga. No se reintentan errores de credenciales,
configuración o fallos ajenos al parseo/contrato. Los retries del SDK están
deshabilitados (`max_retries=0`) para evitar multiplicar solicitudes.

Los logs registran un identificador por ejecución, intentos, tipo de falla y
validación exitosa. Omiten texto, respuestas y claves. No activar tracing externo
con datos privados sin evaluar su política de datos.

Se reutilizan los nombres de variables y la separación de responsabilidades
del Módulo 1. Esta entrega es independiente: usa adaptadores LangChain y no
modifica ni requiere el repositorio del cliente LLM anterior.

## Verificación reproducible

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Las pruebas usan LCEL y el parser real con `httpx.MockTransport` para OpenAI:
primera respuesta con JSON inválido, objeto incompleto o lista vacía y segunda
respuesta válida. También verifican agotamiento de intentos, truncamiento incluso
con objeto parseado, errores no reintentables y entrada inválida. No consumen API.
Las pruebas simuladas no acreditan acceso real a modelos ni calidad semántica.

Validación de esta entrega (5 de noviembre de 2026): **21 pruebas aprobadas**
y `pip check` sin incompatibilidades, con Python 3.12. Se incluye
`requirements-lock.txt` con las versiones verificadas; puede instalarse con
`python -m pip install -r requirements-lock.txt`. Las pruebas se ejecutaron fuera
de la restricción de sockets de asyncio del sandbox de Windows, manteniendo las
llamadas HTTP simuladas. No se ejecutaron solicitudes facturables a proveedores.

## Referencias oficiales

- [Structured output de ChatOpenAI](https://reference.langchain.com/python/langchain-openai/chat_models/ChatOpenAI/with_structured_output)
- [Structured output de ChatAnthropic](https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic/with_structured_output)
- [RunnableRetry](https://reference.langchain.com/python/langchain-core/runnables/retry/RunnableRetry)
