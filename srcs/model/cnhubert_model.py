from GPT_SoVITS.feature_extractor.cnhubert import CNHubert

from model.model import Model


class CnhubertModel(Model):
    ssl_model = CNHubert

    def get_ssl_model(self) -> CNHubert:
        return self.ssl_model

    def init(self):
        ssl_model = CNHubert(self.model_path)
        ssl_model.eval()
        if self.is_half == True:
            ssl_model = ssl_model.half().to(self.device)
        else:
            ssl_model = ssl_model.to(self.device)
        self.ssl_model = ssl_model
