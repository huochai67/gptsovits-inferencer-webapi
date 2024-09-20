import io
from card import Card
from model.bert_model import BertModel
from cardmanager import modelManager
from tts import get_tts_wav

import soundfile as sf


async def synthesize(card: Card, text: str, language: str):
    reference = card.get_reference()

    # Synthesize audio
    synthesis_result = await get_tts_wav(
        sovits_model=card.get_sovits_model(),
        gpt_model=card.get_gpt_model(),
        prompt=reference.prompt,
        bert_model=modelManager.get_bert_model(),
        bert=reference.bert,
        phones=reference.phones,
        refers=reference.refers,
        text=text,
        text_language=language,
    )

    if synthesis_result:
        last_audio_data, last_sampling_rate = synthesis_result
        wav_buf = io.BytesIO()
        wav_buf.name = "file.wav"
        sf.write(wav_buf, last_audio_data, last_sampling_rate)
        wav_buf.seek(0)
        return wav_buf
