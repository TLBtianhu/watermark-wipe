# 模型权重

这个目录在 Git 仓库中只保留本说明文件。`*.pt` / `*.pth` 权重不会被提交，原因：

- 总体积约 **430MB**，超过 GitHub 普通文件建议大小；
- 部分模型有单独的许可条款，不适合随代码一起再分发。

## 下载

在项目根目录运行：

```bat
python download_models.py
```

国内网络如果很慢，可以使用代理：

```bat
python download_models.py --proxy http://127.0.0.1:7897
```

## 模型清单

| 文件名 | 用途 | 大小 | 来源 | 许可 |
| --- | --- | --- | --- | --- |
| `watermark_yolo11x.pt` | 水印/Logo 检测 | 109 MB | [corzent/yolo11x_watermark_detection](https://huggingface.co/corzent/yolo11x_watermark_detection) | 模型卡标注 MIT；基于 Ultralytics YOLO11（AGPL-3.0），作者限定研究/教育用途 |
| `big-lama.pt` | LaMa 图像修复 | 196 MB | [IOPaint / big-lama](https://github.com/Sanster/models/releases/tag/add_big_lama) | LaMa 为 Apache-2.0 |
| `RealESRGAN_x2plus.pth` | 2 倍超分 | 64 MB | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause |
| `RealESRGAN_x4plus.pth` | 4 倍超分 | 64 MB | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD-3-Clause |

## 注意

- 本项目代码使用 Ultralytics YOLO11，因此仓库整体按 **AGPL-3.0** 发布。
- 权重不随仓库分发，请从上述官方地址下载，并遵守各自许可。
- `download_models.py` 会校验文件大小，避免下载到半截文件。
