import hashlib
import time

from result import Ok, Err, Result

DataType = dict | None | str


class ApiResponsed:
    success: bool
    data: DataType

    def __init__(self, success: bool, data: DataType) -> None:
        self.success = success
        self.data = data


def make_response(success: bool, arg: DataType) -> ApiResponsed:
    return ApiResponsed(success=success, data=arg)


def match_result(result: any):
    match result:
        case Ok(content):
            return make_response(True, content)
        case Err(err):
            return make_response(False, err)


def hash_md5(text: str):
    return hashlib.md5(str.encode(text)).hexdigest()


def uuid_time_md5() -> str:
    return hash_md5(str(time.time()))


def read_file(filepath: str) -> Result[str, str]:
    try:
        with open(filepath, "r") as f:
            return Ok(f.read())
    except IOError:
        return Err("ioerror")


def save_file(filepath: str, text: str) -> Result[int, str]:
    try:
        with open(filepath, "x") as f:
            return Ok(f.write(text))
    except IOError:
        return Err("ioerror")


def md5_file(filepath: str) -> Result[str, str]:
    try:
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5()
            while chunk := f.read(8192):
                file_hash.update(chunk)
        return Ok(file_hash.hexdigest())
    except IOError:
        return Err("ioerror")
