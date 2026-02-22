from pydantic import BaseModel, Field

import config


class TicketClassification(BaseModel):
    classe: str = Field(
        min_length=1,
        description="Categoria exata do ticket (deve ser uma das classes fornecidas).",
    )
    justificativa: str = Field(
        min_length=1,
        description="Explicação em 1 a 3 frases do motivo da classificação, citando termos do ticket, em português.",
    )


class JustificationResponse(BaseModel):
    justificativa: str = Field(
        min_length=1,
        max_length=config.JUSTIFICATION_MAX_LENGTH,
        description="Explicação em 1 a 3 frases em português que justifica a classificação, citando termos do ticket.",
    )
