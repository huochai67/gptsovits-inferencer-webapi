from result import Result, Ok, Err
from pydantic import BaseModel
from fastapi import FastAPI
import soundfile as sf
from fastapi.responses import StreamingResponse

from utils2 import make_response, match_result

from cardmanager import CardManager
from ttshelper import synthesize


class TTSArgs(BaseModel):
    text: str


class WebAPIImpl:
    def init(self, api: FastAPI):
        @api.post("/tts")
        async def tts(card: str, args: TTSArgs):
            match CardManager.try_get(cardname=card):
                case Err(err):
                    return make_response(False, err)
                case Ok(card):
                    a = await synthesize(card=card, text=args.text, language="all_zh")
                    return StreamingResponse(a, media_type="audio/wav")
                    # return make_response(True, "Summit")
            return make_response(True, "Summit")

        @api.get("/get_all_cards")
        async def get_all_cards():
            return make_response(True, list(CardManager.get_all().keys()))


WebAPI = WebAPIImpl()
