# ReadMe

# ChiFraud: A Long-term Web Text Dataset for Chinese Fraud Detection

---

This repo contains code & dataset accompanying the paper, ChiFraud: A Long-term Web Text Dataset for Chinese Fraud Detection.

**Dependencies**

This code requires the following requirements:

- Python 3.8+
- Pytorch 2.1.1+   CUDA 12.1   transformers 4.38.2

**Quick Start**

**Prepare the dataset**

```
git clone https://github.com/xuemingxxx/ChiFraud.git
```

**Train and Test Data ---** Chinese Fraud Texts

```
Label_id  Text
0	   玫瑰花茶：味甘微苦，性温，具有行气解郁、活血散瘀的作用。玫瑰花气味芳香，药性平和，既能疏肝理气而解郁，又能...
8	   现在还能买到不用实名制的手机卡吗_【联系微信3179238】【四年老店】【信誉第一】wereretryxsddsjkl
10	 在淘宝上买弩弓的暗语,【加q:56793.***6】安★全☆诚★信【八.年.老.店.】
```

| Label_id | Fraud Category |
| --- | --- |
| 0 | Normal text |
| 1 | Gambling |
| 2 | Whoring |
| 3 | Fake Credentials |
| 4 | Fake Bank Card |
| 5 | Prohibited Drugs |
| 6 | Unauthorized Cash-Out |
| 7 | Unauthorized Certification |
| 8 | Underground Loan |
| 9 | Fake SIM |
| 10 | New |

**Download Pretrain Models**

Bert:https://huggingface.co/google-bert

ChineseBert: https://huggingface.co/ShannonAI/ChineseBERT-base

Qwen: https://huggingface.co/Qwen

LLaMa2: https://huggingface.co/meta-llama

**Training Baselines**