import torch, numpy as np, onnxruntime as ort, os, sys
from transformers import AutoTokenizer, AutoModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
M="/Users/nickchiang/Desktop/Deeplearning/model_bert_colab_v6e"; S=sys.argv[1]
tok=AutoTokenizer.from_pretrained(M)
model=AutoModelForTokenClassification.from_pretrained(M, attn_implementation="eager").eval()
enc=tok(list("鈴木一朗和唐納·川普見了王小明。"),is_split_into_words=True,return_tensors="pt")
class W(torch.nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,input_ids,attention_mask,token_type_ids): return s.m(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids).logits
fp=f"{S}/model_fp32.onnx"
torch.onnx.export(W(model),(enc["input_ids"],enc["attention_mask"],enc["token_type_ids"]),fp,
  input_names=["input_ids","attention_mask","token_type_ids"],output_names=["logits"],
  dynamic_axes={k:{0:"b",1:"s"} for k in ["input_ids","attention_mask","token_type_ids","logits"]},opset_version=17,dynamo=False)
q=f"{S}/model_int8.onnx"
quantize_dynamic(fp,q,weight_type=QuantType.QUInt8)
print("int8 MB",os.path.getsize(q)/1e6)
