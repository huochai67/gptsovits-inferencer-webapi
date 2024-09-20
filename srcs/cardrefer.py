import torch
import librosa
import numpy as np

from model.sovits_model import SoVITSModel
from model.cnhubert_model import CnhubertModel
from model.bert_model import BertModel


class CardRefer:
    refers: any
    phones: any
    bert: any
    prompt : any

    def __init__(
        self,
        sovits_model: SoVITSModel,
        cnhubert_model: CnhubertModel,
        bert_model: BertModel,
        ref_wav_path: str,
        prompt_text: str,
        prompt_language: str,
    ):
        # 间隔语音0.3s
        zero_wav = np.zeros(
            int(sovits_model.hps.data.sampling_rate * 0.3),
            dtype=np.float16 if sovits_model.is_half == True else np.float32,
        )
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)
            if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
                raise OSError("参考音频在3~10秒范围外，请更换！")
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)
            if sovits_model.is_half == True:
                wav16k = wav16k.half().to(sovits_model.device)
                zero_wav_torch = zero_wav_torch.half().to(sovits_model.device)
            else:
                wav16k = wav16k.to(sovits_model.device)
                zero_wav_torch = zero_wav_torch.to(sovits_model.device)
            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_content = (
                cnhubert_model.get_ssl_model()
                .model(wav16k.unsqueeze(0))["last_hidden_state"]
                .transpose(1, 2)
            )  # .float()
            codes = sovits_model.vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]
            prompt = prompt_semantic.unsqueeze(0).to(sovits_model.device)

        phones1, bert1, norm_text1 = bert_model.get_phones_and_bert(
            prompt_text, prompt_language, sovits_model.version
        )

        self.prompt = prompt
        self.bert = bert1
        self.phones = phones1
        self.refers = [
            sovits_model.get_spepc(ref_wav_path).to(sovits_model.dtype).to(sovits_model.device)
        ]
