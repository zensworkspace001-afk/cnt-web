import json, random, torch, numpy as np, onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import f1_score
M="/Users/nickchiang/Desktop/Deeplearning/model_bert_colab_v6e"
tok=AutoTokenizer.from_pretrained(M); model=AutoModelForTokenClassification.from_pretrained(M).eval()
id2=model.config.id2label
import os,sys; sess=ort.InferenceSession(sys.argv[1]+"/"+os.environ.get("QF","model_int8.onnx"))
rows=[json.loads(l) for l in open("/Users/nickchiang/Desktop/Deeplearning/combined_test.jsonl")]
random.seed(0); rows=random.sample(rows,400)
gold,pt,ox=[],[],[]; agree=tot=0
for r in rows:
    enc=tok(r["chars"],is_split_into_words=True,return_tensors="pt",truncation=True,max_length=512)
    wid=enc.word_ids()
    with torch.no_grad(): a=model(**enc).logits[0].argmax(-1).tolist()
    b=sess.run(None,{k:enc[k].numpy() for k in ["input_ids","attention_mask","token_type_ids"]})[0][0].argmax(-1).tolist()
    def al(p):
        t=["O"]*len(r["chars"]); seen=set()
        for pid,w in zip(p,wid):
            if w is None or w in seen: continue
            seen.add(w); t[w]=id2[pid]
        return t
    A,B=al(a),al(b); g=r["tags"][:len(A)]
    gold.append(g); pt.append(A); ox.append(B)
    agree+=sum(x==y for x,y in zip(A,B)); tot+=len(A)
print("tag agreement",agree/tot); print("F1 pytorch",f1_score(gold,pt),"F1 int8",f1_score(gold,ox))
