import re

import torch
import LangSegment
from GPTSoVITS.GPT_SoVITS.text import chinese
from GPTSoVITS.GPT_SoVITS.text import cleaned_text_to_sequence
from GPTSoVITS.GPT_SoVITS.text.cleaner import clean_text
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PretrainedBartModel,
)
from model.model import Model


def clean_text_inf(text, language, version):
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)
    return phones, word2ph, norm_text


class BertModel(Model):
    tokenizer: PreTrainedTokenizer
    bert_model: PretrainedBartModel

    def get_tokenizer(self) -> PreTrainedTokenizer:
        return self.tokenizer

    def get_bert_model(self) -> PreTrainedTokenizer:
        return self.bert_model

    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.get_tokenizer()(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)
            res = self.get_bert_model()(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
        assert len(word2ph) == len(text)
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)
        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        return phone_level_feature.T

    def get_bert_inf(self, phones, word2ph, norm_text, language):
        language = language.replace("all_", "")
        if language == "zh":
            bert = (
                self.get_bert_feature(norm_text, word2ph).to(self.device).to(self.dtype)
            )
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=self.dtype,
            ).to(self.device)

        return bert

    def get_phones_and_bert(self, text, language, version, final=False):
        if language in {"en", "all_zh", "all_ja", "all_ko", "all_yue"}:
            language = language.replace("all_", "")
            if language == "en":
                LangSegment.setfilters(["en"])
                formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
            else:
                # 因无法区别中日韩文汉字,以用户输入为准
                formattext = text
            while "  " in formattext:
                formattext = formattext.replace("  ", " ")
            if language == "zh":
                if re.search(r"[A-Za-z]", formattext):
                    formattext = re.sub(
                        r"[a-z]", lambda x: x.group(0).upper(), formattext
                    )
                    formattext = chinese.mix_text_normalize(formattext)
                    return self.get_phones_and_bert(formattext, "zh", version)
                else:
                    phones, word2ph, norm_text = clean_text_inf(
                        formattext, language, version
                    )
                    bert = self.get_bert_feature(norm_text, word2ph).to(self.device)
            elif language == "yue" and re.search(r"[A-Za-z]", formattext):
                formattext = re.sub(r"[a-z]", lambda x: x.group(0).upper(), formattext)
                formattext = chinese.mix_text_normalize(formattext)
                return self.get_phones_and_bert(formattext, "yue", version)
            else:
                phones, word2ph, norm_text = clean_text_inf(
                    formattext, language, version
                )
                bert = torch.zeros(
                    (1024, len(phones)),
                    dtype=self.dtype,
                ).to(self.device)
        elif language in {"zh", "ja", "ko", "yue", "auto", "auto_yue"}:
            textlist = []
            langlist = []
            LangSegment.setfilters(["zh", "ja", "en", "ko"])
            if language == "auto":
                for tmp in LangSegment.getTexts(text):
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            elif language == "auto_yue":
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "zh":
                        tmp["lang"] = "yue"
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
            else:
                for tmp in LangSegment.getTexts(text):
                    if tmp["lang"] == "en":
                        langlist.append(tmp["lang"])
                    else:
                        # 因无法区别中日韩文汉字,以用户输入为准
                        langlist.append(language)
                    textlist.append(tmp["text"])
            print(textlist)
            print(langlist)
            phones_list = []
            bert_list = []
            norm_text_list = []
            for i in range(len(textlist)):
                lang = langlist[i]
                phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
                bert = self.get_bert_inf(phones, word2ph, norm_text, lang)
                phones_list.append(phones)
                norm_text_list.append(norm_text)
                bert_list.append(bert)
            bert = torch.cat(bert_list, dim=1)
            phones = sum(phones_list, [])
            norm_text = "".join(norm_text_list)

        if not final and len(phones) < 6:
            return self.get_phones_and_bert("." + text, language, version, final=True)

        return phones, bert.to(self.dtype), norm_text

    def init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        bert_model = AutoModelForMaskedLM.from_pretrained(self.model_path)

        self.dtype = torch.float16 if self.is_half == True else torch.float32
        if self.is_half == True:
            self.bert_model = bert_model.half().to(self.device)
        else:
            self.bert_model = bert_model.to(self.device)
