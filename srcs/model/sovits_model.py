from typing import Self

import torch
from GPTSoVITS.GPT_SoVITS.module.models import SynthesizerTrn
from GPTSoVITS.GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPTSoVITS.tools.my_utils import load_audio

from model.model import Model


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


class SoVITSModel(Model):
    vq_model: SynthesizerTrn
    hps: dict
    version: str
    dict_language: dict

    def init(self) -> Self:
        self.dtype = torch.float16 if self.is_half == True else torch.float32

        dict_s2 = torch.load(self.model_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        if dict_s2["weight"]["enc_p.text_embedding.weight"].shape[0] == 322:
            hps.model.version = "v1"
        else:
            hps.model.version = "v2"

        # print("sovits版本:",hps.model.version)

        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        if "pretrained" not in self.model_path:
            del vq_model.enc_q
        if self.is_half == True:
            vq_model = vq_model.half().to(self.device)
        else:
            vq_model = vq_model.to(self.device)
        vq_model.eval()
        print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
        # dict_language = dict_language_v1 if version =='v1' else dict_language_v2
        """
        if prompt_language is not None and text_language is not None:
            if prompt_language in list(dict_language.keys()):
                prompt_text_update, prompt_language_update = {'__type__':'update'},  {'__type__':'update', 'value':prompt_language}
            else:
                prompt_text_update = {'__type__':'update', 'value':''}
                prompt_language_update = {'__type__':'update', 'value':i18n("中文")}
            if text_language in list(dict_language.keys()):
                text_update, text_language_update = {'__type__':'update'}, {'__type__':'update', 'value':text_language}
            else:
                text_update = {'__type__':'update', 'value':''}
                text_language_update = {'__type__':'update', 'value':i18n("中文")}
            return  {'__type__':'update', 'choices':list(dict_language.keys())}, {'__type__':'update', 'choices':list(dict_language.keys())}, prompt_text_update, prompt_language_update, text_update, text_language_update
        """

        self.version = hps.model.version
        self.vq_model = vq_model
        self.hps = hps
        self.dict_language = []

    def get_spepc(self, filename: str):
        audio = load_audio(filename, int(self.hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        maxx = audio.abs().max()
        if maxx > 1:
            audio /= min(2, maxx)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(
            audio_norm,
            self.hps.data.filter_length,
            self.hps.data.sampling_rate,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False,
        )
        return spec
