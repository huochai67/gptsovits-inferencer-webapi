import torch
from GPTSoVITS.GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule

from model.model import Model


class GPTModel(Model):
    hz: int
    max_sec: int
    t2s_model: str
    config: any

    def init(self):
        self.hz = 50

        # load model
        dict_s1 = torch.load(self.model_path, map_location="cpu")
        config = dict_s1["config"]

        # model info
        self.max_sec = config["data"]["max_sec"]
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        if self.is_half == True:
            t2s_model = t2s_model.half()
        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))

        # save model
        self.config = config
        self.t2s_model = t2s_model
