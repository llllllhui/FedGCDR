[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/llllllhui/FedGCDR)
<div align="center">

<h2 class="papername"> Federated Graph Learning for Cross-Domain Recommendation </h2>
<div>
<div>
    Ziqi Yang<sup>1,2</sup>, Zhaopeng Peng<sup>1,2</sup>, Zihui Wang<sup>1,2</sup>, Jianzhong Qi<sup>3</sup>, Chaochao Chen<sup>4</sup>,Weike Pan<sup>5</sup>, Chenglu Wen<sup>1,2</sup>, Cheng Wang<sup>1,2</sup>, Xiaoliang Fan<sup>1,2∗</sup>

</div>

<div><sup>1</sup>Fujian Key Laboratory of Sensing and Computing for Smart Cities, Xiamen University, China</div>
<div><sup>2</sup>Key Laboratory of Multimedia Trusted Perception and Efficient Computing,Ministry of Education of China, Xiamen University, China</div>
<div><sup>3</sup>School of Computing and Information Systems, The University of Melbourne, Australia</div>
<div><sup>4</sup>College of Computer Science and Technology, Zhejiang University Hangzhou, China</div>
<div><sup>5</sup>College of Computer Science and Software Engineering, Shenzhen University Shenzhen, China</div>
<div>*Corresponding author</div>
<br>
<div>

[The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS 2024)](https://neurips.cc/Conferences/2024)

[[Paper]](https://arxiv.org/abs/2410.08249)

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLafinHana%2FFedGCDR&count_bg=%2379C83D&title_bg=%23555555&icon=github.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

</div>
</div>
<br>

</div>



## Introduction

This is the github repository of *Federated Graph Learning for Cross-Domain Recommendation*. In this apper, we propose a federated graph learning framework to preserve intra and inter domain privacy while mitigating negative transfer in cross-domain recommendation.

Our method is consist of two key modules, the positive knowledge transfer module and the positive knowledge activation module. The first module can extract domain konwledge form graph attention networks in a privacy perserving mode and has a theoretical guarantee. The second module performs graph expansion with virtual social links and incorporates a fine-tuning process to filter potientially harmful information and make maximum use of external knowledge.

The architecture of the proposed model:

<div align="center">
<img src='./assets/FedGCDR.png' width='100%'>
</div>

## Benchmark

We conduct experiments on 16 popular domains of the Amazon datasets:

<div align="center">
<img src='./assets/dataset.png' width='100%'>
</div>

## Evaluation results

Here we list the recommendation performance comparison of our method and baselines. Please refer to our paper for more details.

<div align="center">
<img src='./assets/results.png' width='100%'>
</div>


## Citation

If you find this work useful for your research, please kindly cite our paper:
```
@article{yang2024federated,
  title={Federated Graph Learning for Cross-Domain Recommendation},
  author={Yang, Ziqi and Peng, Zhaopeng and Wang, Zihui and Qi, Jianzhong and Chen, Chaochao and Pan, Weike and Wen, Chenglu and Wang, Cheng and Fan, Xiaoliang},
  journal={arXiv preprint arXiv:2410.08249},
  year={2024}
}
```


