from pydantic import AliasChoices, BaseModel, Field


class CommonMixin(BaseModel):
    progress: bool = Field(
        False,
        validation_alias=AliasChoices("P", "progress"),
        description="show progress",
    )
