"""Token ids to text, as the Parakeet tokenizer decodes them with skip_special_tokens=True."""

import json
import re

BYTE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


class Tokenizer:
    def __init__(self, path: str):
        spec = json.load(open(path))
        self.pieces = {i: p for p, i in spec["model"]["vocab"].items()}
        self.special = set()
        for token in spec["added_tokens"]:
            self.pieces[token["id"]] = token["content"]
            if token["special"]:
                self.special.add(token["id"])

    def decode(self, ids: list[int]) -> str:
        out, pending = [], bytearray()
        for i in ids:
            if i in self.special:
                continue
            piece = self.pieces[i]
            if m := BYTE.fullmatch(piece):
                pending.append(int(m.group(1), 16))
                continue
            if pending:
                out.append(pending.decode("utf-8", "replace"))
                pending.clear()
            out.append(piece)
        if pending:
            out.append(pending.decode("utf-8", "replace"))
        return "".join(out).replace("▁", " ").strip()
