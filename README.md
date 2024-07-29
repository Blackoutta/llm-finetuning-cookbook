# LLM微调食谱

## 项目简介
一个特定的LLM如何高效微调？

怎么处理数据集？

超参数都怎么设？

怎么评估训练结果？

本项目旨在通过自己动手微调过的项目来记录这些问题，供快速查阅。
```
.
├── chatglm3【ChatGLM3微调AdvertiseGen数据集)】
│   ├── chatglm3-6b-ft-advertisegen.py      【Python微调脚本】    
│   ├── chatglm3_6b_finetuning.ipynb        【调试微调脚本的notebook】
│   ├── chatglm3_6b_inference.ipynb         【训练完成后做推理的notebook】
│   ├── metrics.py                          【Bleu4模型评估函数】
│   ├── tokenization.py                     【针对数据集的tokenize函数】
│   └── training_log.png 
├── gemma2 【Gemma2微调《鸣潮》游戏私有合成数据集】
│   ├── gemma2_mingchao_ft.ipynb               【微调notebook】
│   ├── gemma2_mingchao_inference_after.ipynb  【微调前进行推理的notebook】
│   ├── gemma2_mingchao_inference_before.ipynb 【微调后进行推理的notebook】
│   └── synthetic_data_merge.jsonl             【使用Self-instruct合成的私有数据集】
├── quantization
│   ├── quant_AWQ-homework.ipynb  【AWQ量化notebook】
│   └── quant_GPTQ_homework.ipynb 【GPTQ量化notebook】

```