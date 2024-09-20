import asyncio

import torch
import numpy as np

from model.manager import GPTModel, SoVITSModel, BertModel


def tts_segment(
    sovits_model: SoVITSModel,
    gpt_model: GPTModel,
    bert_model: BertModel,
    text: str,
    text_language: str,
    prompt: list[torch.Tensor],
    bert1: torch.Tensor,
    phones1: list[int],
    refers: list[torch.Tensor],
    top_k: int,
    top_p: float,
    temperature: float,
    speed: int,
) -> np.ndarray:
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
    with torch.no_grad():
        pred_semantic, idx = gpt_model.t2s_model.model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            prompt,
            bert,
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


def tts_segments(
    sovits_model: SoVITSModel,
    gpt_model: GPTModel,
    bert_model: BertModel,
    prompt: torch.Tensor,
    bert: torch.Tensor,
    phones: list[int],
    refers: list[torch.Tensor],
    texts: list[str],
    text_language: str,
    top_k: int,
    top_p: float,
    temperature: float,
    speed: int,
):
    for text in texts:
        yield tts_segment(
            sovits_model=sovits_model,
            gpt_model=gpt_model,
            bert_model=bert_model,
            text=text,
            prompt=prompt,
            bert1=bert,
            phones1=phones,
            refers=refers,
            speed=speed,
            text_language=text_language,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )


async def tts_segmentwithdata(data: any, *args, **kwargs):
    return data, tts_segment(*args, **kwargs)


async def tts_segments_Concurrency(
    sovits_model: SoVITSModel,
    gpt_model: GPTModel,
    bert_model: BertModel,
    prompt: torch.Tensor,
    bert: torch.Tensor,
    phones: list[int],
    refers: list[torch.Tensor],
    texts: list[str],
    text_language: str,
    top_k: int,
    top_p: float,
    temperature: float,
    speed: int,
) -> list[np.ndarray]:
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
                bert1=bert,
                phones1=phones,
                refers=refers,
                speed=speed,
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
        )
    results = await asyncio.gather(*tasks)
    results.sort(key=lambda x: x[0])
    return map(lambda x: x[1], results)


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


def process_text(text: str) -> list[str]:
    print("实际输入的目标文本:", text)
    texts = text.split("\n")

    for t in texts:
        if t[-1] not in splits:
            t += "."
    print("实际输入的目标文本(切句后):", texts)
    return texts


def gen_wav(
    audios: list[np.ndarray], sampling_rate, dtype, slience_time: float = 0.3
) -> any:
    zero_wav = np.zeros(
        int(sampling_rate * slience_time),
        dtype=dtype,
    )
    audio_opt = []
    for audio in audios:
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
    return (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)


def get_tts_wav(
    sovits_model: SoVITSModel,
    gpt_model: GPTModel,
    bert_model: BertModel,
    prompt: torch.Tensor,
    bert: torch.Tensor,
    phones: list[int],
    refers: list[torch.Tensor],
    text: str,
    text_language: str = "all_zh",
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    speed=1,
):
    return (
        gen_wav(
            audios=list(
                tts_segments(
                    sovits_model=sovits_model,
                    gpt_model=gpt_model,
                    bert_model=bert_model,
                    prompt=prompt,
                    bert=bert,
                    phones=phones,
                    refers=refers,
                    texts=process_text(text=text),
                    text_language=text_language,
                    top_k=top_k,
                    top_p=top_p,
                    temperature=temperature,
                    speed=speed,
                )
            ),
            sampling_rate=sovits_model.hps.data.sampling_rate,
            dtype=np.float16 if sovits_model.is_half == True else np.float32,
        ),
        sovits_model.hps.data.sampling_rate,
    )


async def get_tts_wav_async(
    sovits_model: SoVITSModel,
    gpt_model: GPTModel,
    bert_model: BertModel,
    prompt: torch.Tensor,
    bert: torch.Tensor,
    phones: list[int],
    refers: list[torch.Tensor],
    text: str,
    text_language: str = "all_zh",
    top_k=20,
    top_p=0.6,
    temperature=0.6,
    speed=1,
):
    return (
        gen_wav(
            audios=await tts_segments_Concurrency(
                sovits_model=sovits_model,
                gpt_model=gpt_model,
                bert_model=bert_model,
                prompt=prompt,
                bert=bert,
                phones=phones,
                refers=refers,
                texts=process_text(text=text),
                text_language=text_language,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                speed=speed,
            ),
            sampling_rate=sovits_model.hps.data.sampling_rate,
            dtype=np.float16 if sovits_model.is_half == True else np.float32,
        ),
        sovits_model.hps.data.sampling_rate,
    )
