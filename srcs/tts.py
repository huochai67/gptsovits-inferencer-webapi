import asyncio
from time import time as ttime

import torch
import librosa
import numpy as np

from model.manager import GPTModel, SoVITSModel, BertModel, CnhubertModel

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}


def tts_segment(
    sovits_model: SoVITSModel,
    gpt_model: GPTModel,
    bert_model: BertModel,
    text: str,
    prompt: str,
    bert1,
    phones1,
    refers,
    speed: float,
    text_language,
    top_k=20,
    top_p=0.6,
    temperature=0.6,
) -> any:
    if text[-1] not in splits:
        text += "。" if text_language != "en" else "."
    print("实际输入的目标文本(每句):", text)
    phones2, bert2, norm_text2 = bert_model.get_phones_and_bert(
        text, text_language, sovits_model.version
    )
    print("前端处理后的文本(每句):", norm_text2)
    bert = torch.cat([bert1, bert2], 1)
    all_phoneme_ids = (
        torch.LongTensor(phones1 + phones2).to(gpt_model.device).unsqueeze(0)
    )

    bert = bert.to(gpt_model.device).unsqueeze(0)
    all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(gpt_model.device)
    # cache_key="%s-%s-%s-%s-%s-%s-%s-%s"%(ref_wav_path,prompt_text,prompt_language,text,text_language,top_k,top_p,temperature)
    # print(cache.keys(),if_freeze)
    with torch.no_grad():
        pred_semantic, idx = gpt_model.t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            prompt,
            bert,
            # prompt_phone_len=ph_offset,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=gpt_model.hz * gpt_model.max_sec,
        )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
    audio = (
        sovits_model.vq_model.decode(
            pred_semantic,
            torch.LongTensor(phones2).to(sovits_model.device).unsqueeze(0),
            refers,
            speed=speed,
        )
        .detach()
        .cpu()
        .numpy()[0, 0]
    )
    max_audio = np.abs(audio).max()  # 简单防止16bit爆音
    if max_audio > 1:
        audio /= max_audio
    return audio


async def tts_segmentwithdata(data: any, *args, **kwargs):
    return data, tts_segment(*args, **kwargs)


# 处理整段文字，按\n分割，基本不对文本做处理
async def get_tts_wav(
    sovits_model: SoVITSModel,
    gpt_model: GPTModel,
    bert_model: BertModel,
    cnhubert_model: CnhubertModel,
    ref_wav_path,
    prompt_text,
    text,
    prompt_language: str = "all_zh",
    text_language: str = "all_zh",
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    speed=1,
):
    assert sovits_model.is_half == gpt_model.is_half
    is_half = sovits_model.is_half
    device = sovits_model.device

    print("实际输入的目标文本:", text)

    print("实际输入的目标文本(切句后):", text)
    texts = text.split("\n")
    # texts = merge_short_text_in_array(texts, 5)

    # 间隔语音0.3s
    zero_wav = np.zeros(
        int(sovits_model.hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise OSError("参考音频在3~10秒范围外，请更换！")
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = (
            cnhubert_model.get_ssl_model()
            .model(wav16k.unsqueeze(0))["last_hidden_state"]
            .transpose(1, 2)
        )  # .float()
        codes = sovits_model.vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

    audio_opt = []
    phones1, bert1, norm_text1 = bert_model.get_phones_and_bert(
        prompt_text, prompt_language, sovits_model.version
    )

    refers = [sovits_model.get_spepc(ref_wav_path).to(sovits_model.dtype).to(device)]
    tasks = []
    for i_text, text in enumerate(texts):
        tasks.append(
            tts_segmentwithdata(
                i_text,
                sovits_model=sovits_model,
                gpt_model=gpt_model,
                bert_model=bert_model,
                text=text,
                prompt=prompt,
                bert1=bert1,
                phones1=phones1,
                refers=refers,
                speed=speed,
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        )
        """
        audio = tts_segment(
            sovits_model=sovits_model,
            gpt_model=gpt_model,
            bert_model=bert_model,
            text=text,
            prompt=prompt,
            bert1=bert1,
            phones1=phones1,
            refers=refers,
            speed=speed,
            text_language=text_language,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        """
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    for result in results:
        audio_opt.append(result[1])
        audio_opt.append(zero_wav)
    return (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    ), sovits_model.hps.data.sampling_rate
    """
    yield sovits_model.hps.data.sampling_rate, (
        np.concatenate(audio_opt, 0) * 32768
    ).astype(np.int16)
    """
