# Third-party components / 第三方组件

Third-party licenses apply to their respective materials. No license is added for project-authored code in this demo release.

第三方许可证适用于其对应材料。本次 demo 发布暂不为项目原创代码添加许可证。

| Component / 组件 | Source / 来源 | Terms / 条款 |
| --- | --- | --- |
| MiT-B2 architecture/configuration and pretrained backbone / MiT-B2 架构、配置及预训练主干 | [nvidia/mit-b2](https://huggingface.co/nvidia/mit-b2) | [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE); local copy: `licenses/SegFormer-LICENSE.txt` |
| Frozen CLIP ViT-L/14, timm format / 冻结 CLIP 视觉编码器 | [timm/vit_large_patch14_clip_224.openai](https://huggingface.co/timm/vit_large_patch14_clip_224.openai), revision `18d0535469bb561bf468d76c1d73aa35156c922b` | Model card lists Apache-2.0; local copy: `licenses/timm-Apache-2.0.txt` |
| Original OpenAI CLIP / 原始 OpenAI CLIP | [OpenAI CLIP](https://github.com/openai/CLIP) | MIT; local copy: `licenses/CLIP-MIT-LICENSE.txt` |

The downloadable CLIP checkpoint contains the vision encoder in native timm format; the text encoder is omitted. The trained student checkpoints contain the fine-tuned MiT-B2 backbone. The GitHub repository distributes code, configuration, and the supplied paper figures; model tensors are provided through the separate download linked in README.md.

网盘 CLIP 权重为 timm 原生格式的视觉编码器，不含文本编码器。学生模型权重包含微调后的 MiT-B2 主干。GitHub 仓库提供代码、配置与论文配图，模型张量通过 README.md 中的独立网盘链接提供。

The SegFormer license includes research/non-commercial use limitations for the covered materials. Those upstream terms remain applicable. Dataset files are not bundled. Python dependencies retain their respective licenses.

SegFormer 许可证对其覆盖材料包含研究/非商业用途限制，上游条款继续适用。本仓库不包含数据集；Python 依赖保留各自许可证。
