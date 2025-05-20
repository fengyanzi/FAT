# FAT (IEEE TGRS 2025)

## Mitigating Texture Bias: A Remote Sensing Super-Resolution Method Focusing on High-Frequency Texture Reconstruction

## 抑制纹理偏见： 一种聚焦于高频纹理重建的遥感图像超分辨率方法


This repository is an official implementation of the paper "*[Mitigating Texture Bias: A Remote Sensing Super-Resolution Method Focusing on High-Frequency Texture Reconstruction](https://ieeexplore.ieee.org/document/10912673)*"

### :tada::tada: News :tada::tada:
- **2025/5/20**  ✅ 代码完整发布   **Code fully released**  
  *(5/18刚完成本科毕设与新论文，是谁520没有女朋友却在疯狂润色代码写说明？原来是我这个小丑🤗)*  
  *(Just finished undergrad thesis and new paper on 5/18. Who's coding frantically on 520 without a girlfriend? Oh it's me, the clown 🤗)* 
- **2025/4/8**  🚀 模型代码公开    **Model code released**
- **2025/3/2** 🌍 RSSR25数据集登陆HuggingFace 🤗   **RSSR25🤗 now on [HuggingFace](https://huggingface.co/datasets/fengyanzi/RSSR25/tree/main)**  
-  **2025/2/27** ✨ 双喜临门！  **Double celebration!**  论文被IEEE TGRS接收 🎉  
    *Paper accepted by [IEEE TGRS](https://ieeexplore.ieee.org/document/10912673) 
    🎉* 同日另一篇论文被CVPR 2025接收 🏆      *Another paper accepted by [CVPR 2025](https://arxiv.org/abs/2504.09621) on the same day 🏆*
- **2025/1/2**  🔍 论文获得大修决定    **Paper received major revision decision**  
  📝 正在积极修改中 / **Actively revising**

说明在进一步完善中

<!-- ### :tada::tada: News :tada::tada:
- 2025/5/20 The code was full released. （5/18刚刚完成本科毕设与新paper，到底是谁520没有女朋友陪着却在猛写代码，原来是我这个小丑🤗）
- 2025/4/8 The model code was released.
- 2025/3/2 RSSR25🤗[huggingface](https://huggingface.co/datasets/fengyanzi/RSSR25/tree/main) 数据集被公开
- 2025/2/27 该论文被TGRS接受:tada:  同一天我们的另外一篇论文被CVPR2025接受
- 2025/1/2 该论文获得大修决定 -->





- [x] Upload the code of FAT
- [x] Upload the dataset RSSR25
- [x] Upload the code of LAM
- [x] Upload the introduction



Paper link [https://ieeexplore.ieee.org/document/10912673](https://ieeexplore.ieee.org/document/10912673)

However, another paper was accepted by CVPR, we need some time to deal with the paper, code, dataset, and undergraduate graduation project. 

RSSR25 Datasets: [BaiduCloud](https://pan.baidu.com/s/1Ywy6W6eVLsJ7nVVoKf6HaQ?pwd=4321) 🤗[huggingface](https://huggingface.co/datasets/fengyanzi/RSSR25/tree/main)
 
Additionally, I provide a user-friendly [LAM](https://github.com/fengyanzi/Local-Attribution-Map-for-Super-Resolution) diagnostic tool, that can be run in 2024.

The Local Attribution Map (LAM) is an interpretability tool designed for super-resolution tasks. It identifies the pixels in a low-resolution input image that have the most significant impact on the network’s output. By analyzing the local regions of the super-resolution result, LAM highlights the areas contributing the most to the reconstruction, providing insights into the model's decision-making process.

Currently, I can offer some super-resolution example images.

![img](./docx/test.png)


## 📖 Citation

If you find our code useful, please consider citing our paper:

```
@article{yan2025mitigating,
  title={Mitigating texture bias: A remote sensing super-resolution method focusing on high-frequency texture reconstruction},
  author={Yan, Xinyu and Chen, Jiuchen and Xu, Qizhi and Li, Wei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025},
  publisher={IEEE}
}
```
