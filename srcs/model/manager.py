import os
from pathlib import Path
from model.bert_model import BertModel
from model.cnhubert_model import CnhubertModel
from model.gpt_model import GPTModel
from model.sovits_model import SoVITSModel

import torch

supportcuda = torch.cuda.is_available()


class ModelManager:
    gpt_models = {}
    sovits_models = {}
    bert_model: BertModel | None = None
    cnhubert_model: CnhubertModel | None = None

    data_dir: str
    use_cuda: bool
    device: str
    dtype = None

    def get_bert_model(self) -> BertModel:
        if not self.bert_model:
            self.bert_model = BertModel(
                model_path=f"{self.data_dir}/chinese-roberta-wwm-ext-large",
                is_half=self.use_cuda,
                device=self.device,
            )
        return self.bert_model

    def get_cnhubert_model(self) -> CnhubertModel:
        if not self.cnhubert_model:
            self.cnhubert_model = CnhubertModel(
                model_path=f"{self.data_dir}/chinese-hubert-base",
                is_half=self.use_cuda,
                device=self.device,
            )
        return self.cnhubert_model

    def get_from_dict(self, key, container) -> any:
        if key in container:
            return container[key]

    def get_gpt_model(self, modelname: str) -> GPTModel:
        if not modelname in self.gpt_models:
            model = GPTModel(
                f"{self.data_dir}/gpt_models/{modelname}.ckpt", self.use_cuda, self.device
            )
            self.gpt_models[modelname] = model
        return self.gpt_models[modelname]

    def get_sovits_model(self, modelname: str) -> SoVITSModel:
        if not modelname in self.sovits_models:
            model = SoVITSModel(
                f"{self.data_dir}/sovits_models/{modelname}.pth", self.use_cuda, self.device
            )
            self.sovits_models[modelname] = model
        return self.sovits_models[modelname]

    def __init__(self, data_dir: str, use_cuda: bool = True, lazyload: bool = True):
        self.use_cuda = supportcuda and use_cuda
        self.device = "cuda" if use_cuda else "cpu"
        self.dtype = torch.float16 if use_cuda == True else torch.float32
        self.data_dir = data_dir

        entrys = os.listdir(data_dir)
        if not (
            "gpt_models"
            and "sovits_models"
            and "chinese-hubert-base"
            and "chinese-roberta-wwm-ext-large" in entrys
        ):
            raise IOError("")

        if lazyload:
            self.get_bert_model()
            self.get_cnhubert_model()

            for gpt_model in Path(f"{data_dir}/gpt_models").iterdir():
                self.get_gpt_model(gpt_model.stem)
            for sovits_model in Path(f"{data_dir}/sovits_models").iterdir():
                self.get_sovits_model(sovits_model.stem)
