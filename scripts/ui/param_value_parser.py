from typing import Any, Dict

from modules.scripts import script_callbacks


class ParamValueParser:
    value_types: Dict[str, type] = {}

    @staticmethod
    def update(infotext: str, params: Dict[str, Any]):
        for key, value_type in ParamValueParser.value_types.items():
            value = params.get(key, None)
            if value is None:
                return
            if value_type == list and not isinstance(value, list):
                params[key] = value.split(";")

    @staticmethod
    def add(key: str, value_type: type):
        if len(ParamValueParser.value_types) == 0:
            script_callbacks.on_infotext_pasted(ParamValueParser.update)
        ParamValueParser.value_types[key] = value_type
