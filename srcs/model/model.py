class Model:
    model_path: str
    is_half: bool
    device: str
    dtype = None

    def __init__(self, model_path: str, is_half: bool, device: str):
        self.model_path = model_path
        self.is_half = is_half
        self.device = device
        self.init()
        pass
    
    def init():
        pass

    def get_path(self):
        return self.model_path
