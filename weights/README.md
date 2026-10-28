# Model weights / 模型权重

Download **C2Dweights** from [Baidu Netdisk / 百度网盘](https://pan.baidu.com/s/1JAV7ZbY1vyrR0dWVTV7tdw?pwd=6pxg).

Extraction code / 提取码：`6pxg`

Place the three downloaded checkpoints directly in this directory. If the archive extracts into an outer `C2Dweights` folder, copy the checkpoint files inside it here.

将下载的三个权重文件直接放入本目录。如果解压后外层文件夹名为 `C2Dweights`，请将里面的权重文件复制到这里。

```text
weights/
├── c2d_256.pth
├── c2d_512.pth
├── clip_visual.pth
└── mit_b2/
    └── config.json   # already included / 仓库已提供
```

MiT-B2 pretrained tensors are not needed as a separate download for inference: the trained student checkpoints restore the complete backbone. Model weights retain the third-party terms described in [THIRD_PARTY.md](../THIRD_PARTY.md).

推理无需另外下载 MiT-B2 预训练张量：学生模型权重会恢复完整主干。模型权重继续遵守 [THIRD_PARTY.md](../THIRD_PARTY.md) 中说明的第三方条款。
