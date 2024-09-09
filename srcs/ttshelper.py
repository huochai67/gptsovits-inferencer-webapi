import io
from fs import getdatadir
from card import Card
from model.manager import ManagerImpl
from tts import get_tts_wav

import soundfile as sf

ModelManager = ManagerImpl(f"{getdatadir()}/models")


async def synthesize(card: Card, text: str, language: str):
    gpt_model = ModelManager.get_gpt_model(card.gpt_model)
    sovits_model = ModelManager.get_sovits(card.sovits_model)
    ref_audio_path = f"{getdatadir()}/references/{card.reference_audio}"
    ref_text = card.reference_text
    ref_language = card.reference_language

    output_path = "test.wav"
    # Synthesize audio
    synthesis_result = await get_tts_wav(
        sovits_model=sovits_model,
        gpt_model=gpt_model,
        bert_model=ModelManager.get_bert_model(),
        cnhubert_model=ModelManager.get_cnhubert_model(),
        ref_wav_path=ref_audio_path,
        prompt_text=ref_text,
        prompt_language=ref_language,
        text=text,
        text_language=language,
    )

    if synthesis_result:
        last_audio_data, last_sampling_rate = synthesis_result
        wav_buf = io.BytesIO()
        wav_buf.name = 'file.wav'
        sf.write(output_path, last_audio_data, last_sampling_rate)
        wav_buf.seek(0)
        return wav_buf.read()
        print(f"Audio saved to {output_path}")
