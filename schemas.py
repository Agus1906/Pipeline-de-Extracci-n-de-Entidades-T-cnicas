"""Contrato de extracción técnica, independiente del proveedor."""
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Criticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class EntidadesTecnicas(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tecnologias: list[str] = Field(min_length=1, max_length=100)
    nivel_de_criticidad: Criticidad
    resumen_tecnico: str = Field(min_length=10, max_length=2000)

    @field_validator("tecnologias")
    @classmethod
    def validar_tecnologias(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 100 for value in normalized):
            raise ValueError("Cada tecnología debe tener entre 1 y 100 caracteres.")
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("Las tecnologías no deben repetirse.")
        return normalized
