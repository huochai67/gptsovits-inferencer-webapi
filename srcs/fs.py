import os


def getdatadir() -> str:
    return datadir


def setdatadir(path: str):
    global datadir
    datadir = path


def from__file__(path: str):
    setdatadir(os.path.abspath(os.path.dirname(os.path.realpath(path)) + "/../data"))
