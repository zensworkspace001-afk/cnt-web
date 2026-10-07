"""把 tokenizer 單字詞表、姓氏字典、標籤表塞進 index.html 的 /*__DATA__*/ 位置。
用法（在主專案根目錄執行）：python3 web_app/tools/inline_data.py"""
import json, re, sys
sys.path.insert(0, ".")
import surnames
import foreign_names_pool as fp
M = "model_bert_colab_v6e"
tj = json.load(open(f"{M}/tokenizer.json"))
items = sorted(((k, i) for k, i in tj["model"]["vocab"].items() if len(k) == 1), key=lambda x: x[1])
v = tj["model"]["vocab"]
data = {
    "chars": "".join(k for k, _ in items), "ids": [i for _, i in items],
    "special": {k: v[k] for k in ["[CLS]", "[SEP]", "[UNK]", "[PAD]"]},
    "compound": surnames.ALL_COMPOUND, "single": "".join(surnames.ALL_SINGLE),
    # 規則版音譯人名用字：外文人名詞庫裡出現過的字 + 常見音譯用字
    "forChars": "".join(sorted(set("".join(a for cat in fp.CATEGORIES.values() for p in cat for a in p["aliases"]).replace("·", "")
                 + "阿埃艾安奧巴貝比波布查達德迪杜多厄法菲弗福加蓋戈格哈海赫亨胡吉基卡凱坎科克庫拉萊蘭朗勞雷里利林隆盧魯羅洛馬邁曼梅門米莫穆納內尼諾帕佩皮普齊奇恰喬切琴薩塞森沙舍什史斯索蘇塔泰坦特提湯圖托瓦威維溫沃烏西希謝休雅亞楊伊尤約扎澤詹茲茨爾夫伯翰娜莉絲瑪蒂琳珊黛露戴"))),
    "labels": [json.load(open(f"{M}/config.json"))["id2label"][str(i)] for i in range(9)],
}
line = "const DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";"
p = "web_app/index.html"
html = open(p, encoding="utf-8").read()
html = re.sub(r"/\*__DATA__\*/|const DATA = .*;", lambda _: line, html, count=1)
open(p, "w", encoding="utf-8").write(html)
print("ok", len(line) // 1000, "KB")
