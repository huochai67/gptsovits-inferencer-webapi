import json
from typing import Self

from fs import getdatadir


class Card:
    name: str
    gpt_model: str
    sovits_model: str
    reference_audio: str
    reference_text: str
    reference_language: str

    def __init__(
        self,
        name: str,
        gpt_model: str,
        sovits_model: str,
        reference_audio: str,
        reference_text: str,
        reference_language: str,
    ) -> None:
        self.name = name
        self.gpt_model = gpt_model
        self.sovits_model = sovits_model
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.reference_language = reference_language

    def get_gpt_model_path(self) -> str:
        return f"{getdatadir()}/data/models/gpt_models/{self.gpt_model}.ckpt"

    def get_sovits_model_path(self) -> str:
        return f"{getdatadir()}/data/models/sovits_models/{self.sovits_model}.pth"

    def get_reference_audio_path(self) -> str:
        return f"{getdatadir()}/data/references/{self.reference_audio}"

    def from_dict(d: dict) -> Self:
        return Card(
            name=d["name"],
            gpt_model=d["gpt_model"],
            sovits_model=d["sovits_model"],
            reference_audio=d["reference_audio"],
            reference_text=d["reference_text"],
            reference_language=d["reference_language"],
        )

    def from_file(filepath: str) -> Self:
        with open(filepath, "r") as file:
            return Card.from_dict(json.load(file))
