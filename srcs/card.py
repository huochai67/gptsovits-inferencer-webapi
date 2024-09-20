import json
from typing import Self

from fs import getdatadir
from cardrefer import CardRefer

from model.manager import GPTModel, SoVITSModel, BertModel, CnhubertModel, ModelManager


class Card:
    name: str
    gpt_model_name: str
    sovits_model_name: str

    gpt_model: GPTModel = None
    sovits_model: SoVITSModel = None

    reference: CardRefer = None
    modelManger: ModelManager

    def get_gpt_model_path(self) -> str:
        return f"{getdatadir()}/models/gpt_models/{self.gpt_model_name}.ckpt"

    def get_sovits_model_path(self) -> str:
        return f"{getdatadir()}/models/sovits_models/{self.sovits_model_name}.pth"

    def get_ref_audio_path(self) -> str:
        return f"{getdatadir()}/references/{self.reference_audio}"

    def get_gpt_model(self) -> GPTModel:
        if not self.gpt_model:
            self.gpt_model = self.modelManger.get_gpt_model(self.gpt_model_name)
        return self.gpt_model

    def get_sovits_model(self) -> GPTModel:
        if not self.sovits_model:
            self.sovits_model = self.modelManger.get_sovits_model(self.sovits_model_name)
        return self.sovits_model

    def get_reference(self) -> CardRefer:
        if not self.reference:
            self.reference = CardRefer(
                sovits_model=self.get_sovits_model(),
                cnhubert_model=self.modelManger.get_cnhubert_model(),
                bert_model=self.modelManger.get_bert_model(),
                ref_wav_path=self.get_ref_audio_path(),
                prompt_text=self.reference_text,
                prompt_language=self.reference_language,
            )
        return self.reference

    def __init__(
        self,
        name: str,
        gpt_model: str,
        sovits_model: str,
        reference_audio: str,
        reference_text: str,
        reference_language: str,
        modelManger: ModelManager,
        lazyload: bool,
    ):
        self.name = name
        self.gpt_model_name = gpt_model
        self.sovits_model_name = sovits_model
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.reference_language = reference_language
        self.modelManger = modelManger
        if lazyload:
            self.get_reference()

    def from_dict(d: dict, modelManger: ModelManager, lazyload: bool) -> Self:
        return Card(
            name=d["name"],
            gpt_model=d["gpt_model"],
            sovits_model=d["sovits_model"],
            reference_audio=d["reference_audio"],
            reference_text=d["reference_text"],
            reference_language=d["reference_language"],
            modelManger=modelManger,
            lazyload=lazyload,
        )

    def from_file(
        filepath: str, modelManger: ModelManager, lazyload: bool = True
    ) -> Self:
        with open(filepath, "r") as file:
            return Card.from_dict(json.load(file), modelManger, lazyload)
