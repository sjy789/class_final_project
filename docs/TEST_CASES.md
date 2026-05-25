# 测试案例设计

## 已内置案例

`samples/room_background.png`：室内背景。

`samples/plant_foreground.png`：带透明通道的植物前景。

运行：

```powershell
$env:TEMP = "C:\Users\21249\Desktop\作业\机器学习\课程项目\tmp"
$env:TMP = "C:\Users\21249\Desktop\作业\机器学习\课程项目\tmp"
python scripts/smoke_test.py
```

期望输出包括：

- 自动推荐 Top-1 分数和标签。
- 手动评估分数和标签。
- `outputs/<时间戳>/top_composite.png`
- `outputs/<时间戳>/explanation_heatmap.png`
- `outputs/<时间戳>/result.json`

## 后续报告建议案例

1. 合理放置：物体位于地面或桌面附近，大小适中，背景纹理较干净。
2. 边界失败：物体太靠近画面边界或越界。
3. 尺寸失败：物体过大遮挡主体区域，或过小不符合视觉比例。
4. 悬空失败：物体底部缺少支撑区域。
5. 光照不协调：前景和背景亮度差异明显，观察颜色协调前后差异。
6. 背景复杂：候选位置覆盖高纹理区域，模型评分下降。

