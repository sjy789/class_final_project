# 智能物体放置助手

这是课程项目方向 A 的 Web 应用原型：给定背景图和前景物体，系统自动推荐更合理的放置位置，并输出评分、Top-K 候选、合成图和解释热力图。

## 技术栈

- 前端：React + Vite + TypeScript
- 后端：FastAPI
- 图像处理：Pillow + OpenCV
- 本地推理：PyTorch CPU

## 启动方式

首次运行先安装后端依赖：

```powershell
python -m pip install -r backend\requirements.txt
```

前端需要 Node.js。如果项目内存在 `tools\node-v24.16.0-win-x64`，启动脚本会自动使用；否则请先安装 Node.js LTS。

后端：

```powershell
.\scripts\start_backend.ps1
```

前端：

```powershell
.\scripts\start_frontend.ps1
```

浏览器打开：

```text
http://127.0.0.1:5173
```

## 快速验证

```powershell
$env:TEMP = "C:\Users\21249\Desktop\作业\机器学习\课程项目\tmp"
$env:TMP = "C:\Users\21249\Desktop\作业\机器学习\课程项目\tmp"
python scripts/smoke_test.py
```

验证通过后，`outputs/` 会生成合成图、解释热力图和结果 JSON。

## OPA 小模型训练

训练流程见 `docs/TRAINING_OPA_TINY.md`。

项目已包含：

- OPA 风格小模型：`backend/app/opa_cnn.py`
- OPA 数据集读取：`training/opa_dataset.py`
- 训练脚本：`training/train_opa_tiny.py`
- checkpoint 加载与 fallback：`backend/app/trained_scorer.py`

训练完成后，把 `opa_tiny.pt` 放到：

```text
backend/checkpoints/opa_tiny.pt
```

后端会在本地 CPU 上加载该 checkpoint 参与评分；没有 checkpoint 时自动使用 fallback 评分逻辑。

## 项目结构

```text
backend/          FastAPI 接口、图像流水线、评分模型
frontend/         React 交互界面
scripts/          示例数据生成、启动脚本、快速验证脚本
samples/          内置测试图片
outputs/          推理输出结果
docs/             设计说明和测试案例说明
tools/            项目内 Node.js
```

## 功能对应评分点

- 基础模型小改动：`backend/app/model.py`
- 模型改造：候选位置排序和 Top-K 推荐
- 多模型串联：mask、候选生成、评分、颜色协调、合成
- 模型解释：遮挡敏感性热力图
- 复杂交互：上传、拖拽、缩放、Top-K、热力图、导出

## 参考代码定位

参考关系见 `docs/REFERENCE_MAPPING.md`。

本项目明确参考了老师方向 A 给出的三个 BCMI 资源：

- `bcmi/Object-Placement-Assessment-Dataset-OPA`：参考 OPA/SimOPA 的 `RGB 合成图 + foreground mask -> 合理性评分` 输入输出形式。
- `bcmi/TopNet-Object-Placement`：参考多位置、多尺度候选评分和 Top-K 推荐的组织方式。
- `bcmi/libcom`：参考图像合成工具链的模块串联方式，包括前景处理、放置评分、协调处理、合成和热力图展示。

需要诚实说明：当前项目没有直接加载这些仓库的预训练权重，也没有复现完整训练流程；代码采用轻量本地 CPU 适配实现，便于稳定演示。
