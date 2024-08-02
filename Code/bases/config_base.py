import json
from pprint import pprint
import os


class JsonConfig:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.demo_cfg = None

    def get_config(self):
        return self.cfg

    @classmethod
    def save_json_config(cls, cfg, path):
        with open(path, "w", encoding="utf8") as f:
            json.dump(cfg, f, ensure_ascii=False)

    def get_suggested_demo_config_name(self):
        return "{}Demo.json".format(self.__class__.__name__)

    @classmethod
    def load_config(cls, path=None):
        with open(path, "r", encoding="utf8") as f:
            cfg = json.load(f)
        return cfg

    def save_config(self, path=None):
        with open(path, "w", encoding="utf8") as f:
            json.dump(self.cfg, f, ensure_ascii=False)

    def get_demo_config(self):
        return self.demo_cfg

    def show_demo(self):
        pprint(self.demo_cfg)

    def save_demo_config(self, save_dir=None):
        with open(
            os.path.join(save_dir, self.get_suggested_demo_config_name()),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(self.demo_cfg, f, ensure_ascii=False)
