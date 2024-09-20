import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import fs

fs.from__file__(__file__)

import fixsys
from utils2 import make_response
from webapi import WebAPI


app = FastAPI(
    title="GPT-SoVITS-Inference-Webapi",
    description="Inferencer for GPT-SoVITS",
    summary="Powered by Nicefish4520",
    version="0.9dev",
    contact={
        "name": "NiceFish4520",
        "email": "kcass774@gmail.com",
    },
    license_info={
        "name": "GPLv3",
        "url": "https://www.gnu.org/licenses/gpl-3.0.en.html#license-text",
    },
)

origins = [
    "http://localhost",
    "http://localhost:5173",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
WebAPI.init(api=app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
