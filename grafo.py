"""Flujo de consultas de VRAMind con Pydantic Graph."""

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Literal, cast
import unicodedata

from pydantic_ai import Agent, Conversation
from pydantic_graph import GraphBuilder, StepContext, TypeExpression

RutaConsulta = Literal[
    "arquitecturas",
    "hardware",
    "huggingface",
    "web",
    "combinada",
    "general",
]


@dataclass
class EntradaConsulta:
    mensaje: str
    conversacion: Conversation | None = None


@dataclass
class EstadoConsulta:
    entrada: EntradaConsulta | None = None
    ruta: RutaConsulta = "general"
    conversacion: Conversation | None = None


@dataclass(frozen=True)
class ResultadoConsulta:
    texto: str
    ruta: RutaConsulta
    conversacion: Conversation


def clasificar_consulta(mensaje: str) -> RutaConsulta:
    """Clasifica por indicios locales, sin llamar otro modelo."""
    normalizado = unicodedata.normalize("NFD", mensaje.lower())
    normalizado = "".join(caracter for caracter in normalizado if unicodedata.category(caracter) != "Mn")

    def contiene_termino(*terminos: str) -> bool:
        return any(re.search(rf"\b{re.escape(termino)}\b", normalizado) for termino in terminos)

    intenciones = {
        "arquitecturas": bool(re.search(
            r"\b(pascal|turing|ampere|ada lovelace|blackwell|arquitectura|cuda|tensor cores?|rt cores?|dlss)\b",
            normalizado,
        )),
        "hardware": contiene_termino(
            "vram", "ram", "cpu", "hardware", "computadora", "equipo",
            "diagnostico", "diagnosticar", "ejecutar local", "ejecutar modelos", "mi pc",
        ),
        "huggingface": contiene_termino(
            "hugging face", "huggingface", "hf hub", "repositorio del modelo",
        ),
        "web": contiene_termino(
            "reciente", "actualidad", "ultimo", "ultima", "noticia", "novedad",
            "anuncio", "lanzamiento", "precio de hoy",
        ),
    }
    detectadas = [ruta for ruta, presente in intenciones.items() if presente]
    if not detectadas:
        return "general"
    if len(detectadas) > 1:
        return "combinada"
    return cast(RutaConsulta, detectadas[0])


INSTRUCCIONES_RUTA: dict[RutaConsulta, str] = {
    "arquitecturas": (
        "El flujo prioriza documentación NVIDIA. Usa buscar_documentos para datos específicos "
        "o comparaciones verificables; no fuerces RAG para una explicación general."
    ),
    "hardware": (
        "El flujo prioriza el diagnóstico de hardware. Pide solo GPU, VRAM, CPU o RAM que falten "
        "y usa diagnosticar_hardware cuando estén disponibles."
    ),
    "huggingface": (
        "El flujo prioriza la consulta pública de Hugging Face: usa buscar_modelos_hf para buscar "
        "o consultar_modelo_hf para un repositorio concreto."
    ),
    "web": (
        "El flujo prioriza información reciente: usa buscar_web_tavily y conserva títulos, enlaces "
        "y la distinción entre fuentes oficiales y secundarias."
    ),
    "combinada": (
        "El flujo detectó varias necesidades. Atiende cada parte y usa únicamente las tools "
        "pertinentes; puedes usar más de una capability si la consulta lo requiere."
    ),
    "general": (
        "El flujo no detectó una tarea que requiera una capability. Responde directamente y no "
        "uses tools salvo que la pregunta pida datos técnicos específicos o verificables."
    ),
}


def construir_grafo(agente: Agent[Any]):
    """Construye el flujo que clasifica, orienta una ejecución y forma la respuesta."""
    flujo = GraphBuilder(
        name="flujo_consultas_vramind",
        state_type=EstadoConsulta,
        input_type=EntradaConsulta,
        output_type=ResultadoConsulta,
    )

    @flujo.step
    async def clasificar(ctx: StepContext[EstadoConsulta, None, EntradaConsulta]) -> RutaConsulta:
        ctx.state.entrada = ctx.inputs
        ctx.state.ruta = clasificar_consulta(ctx.inputs.mensaje)
        return ctx.state.ruta

    async def ejecutar(ctx: StepContext[EstadoConsulta, None, RutaConsulta]) -> str:
        entrada = ctx.state.entrada
        if entrada is None:
            raise RuntimeError("El grafo no recibió una consulta.")
        resultado = await agente.run(
            entrada.mensaje,
            conversation=deepcopy(entrada.conversacion),
            instructions=INSTRUCCIONES_RUTA[ctx.inputs],
        )
        ctx.state.conversacion = resultado.conversation
        return resultado.output

    def crear_nodo_ruta(ruta: RutaConsulta, identificador: str, etiqueta: str):
        async def ejecutar_ruta(ctx: StepContext[EstadoConsulta, None, RutaConsulta]) -> str:
            if ctx.inputs != ruta:
                raise RuntimeError("El grafo seleccionó una ruta inconsistente.")
            return await ejecutar(ctx)

        return flujo.step(ejecutar_ruta, node_id=identificador, label=etiqueta)

    ejecutar_arquitecturas = crear_nodo_ruta(
        "arquitecturas", "ejecutar_arquitecturas", "Ejecutar ruta NVIDIA"
    )
    ejecutar_hardware = crear_nodo_ruta(
        "hardware", "ejecutar_hardware", "Ejecutar ruta de hardware"
    )
    ejecutar_huggingface = crear_nodo_ruta(
        "huggingface", "ejecutar_huggingface", "Ejecutar ruta Hugging Face"
    )
    ejecutar_web = crear_nodo_ruta("web", "ejecutar_web", "Ejecutar ruta web")
    ejecutar_combinada = crear_nodo_ruta(
        "combinada", "ejecutar_combinada", "Ejecutar rutas combinadas"
    )
    ejecutar_general = crear_nodo_ruta(
        "general", "ejecutar_general", "Ejecutar consulta general"
    )

    @flujo.step
    async def construir_respuesta(ctx: StepContext[EstadoConsulta, None, str]) -> ResultadoConsulta:
        if ctx.state.conversacion is None:
            raise RuntimeError("El agente no devolvió el estado conversacional.")
        return ResultadoConsulta(
            texto=ctx.inputs,
            ruta=ctx.state.ruta,
            conversacion=ctx.state.conversacion,
        )

    decision = (
        flujo.decision(note="Selecciona una ruta sin otra llamada al modelo.")
        .branch(flujo.match(TypeExpression[Literal["arquitecturas"]]).to(ejecutar_arquitecturas))
        .branch(flujo.match(TypeExpression[Literal["hardware"]]).to(ejecutar_hardware))
        .branch(flujo.match(TypeExpression[Literal["huggingface"]]).to(ejecutar_huggingface))
        .branch(flujo.match(TypeExpression[Literal["web"]]).to(ejecutar_web))
        .branch(flujo.match(TypeExpression[Literal["combinada"]]).to(ejecutar_combinada))
        .branch(flujo.match(TypeExpression[Literal["general"]]).to(ejecutar_general))
    )
    flujo.add(
        flujo.edge_from(flujo.start_node).to(clasificar),
        flujo.edge_from(clasificar).to(decision),
        flujo.edge_from(
            ejecutar_arquitecturas,
            ejecutar_hardware,
            ejecutar_huggingface,
            ejecutar_web,
            ejecutar_combinada,
            ejecutar_general,
        ).to(construir_respuesta),
        flujo.edge_from(construir_respuesta).to(flujo.end_node),
    )
    return flujo.build()
