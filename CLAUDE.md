# forge-viewer

机器人仿真场景查看器，核心是自研的现代 OpenGL 渲染器 **forge**。

完整规格在 `/Users/acrl/Projects/prompt/`（01～14）。本机实测的 GL 事实在
`docs/PLATFORM.md`——**那里的每一条都跑过，规格里的数字来自另一台机器，冲突时以
`docs/PLATFORM.md` 为准，并把差异写进去**。

## 铁律

1. **判据必须在移除修复时变红**（12 §12.1）。写完一条判据就去掉修复跑一遍；不红的判据
   什么都没守住。三种典型的假绿：自洽的符号、自证的等式（两边来自同一个常量）、
   只看局部量。
2. **渲染类判据必须走窗口路径**（12 §12.2）。`capture()` 在 numpy 里翻了一次 Y，
   走它去判"画面对不对"曾让整个画面上下颠倒、60 多条测试全绿、两天没人发现。
3. **热路径不分配**（04 §4.3）。连 `reshape` 产生的小 ndarray 对象也算。
4. **不静默失败**。命令失败要给一句人话；后端不支持的开关要标出来；`caps` 说了不支持
   就明确报，不要 `getattr` 之后碰运气。
5. **注释解释"为什么"，不解释"是什么"**。中文，严肃、精准、简洁。

## 分层

```
ui/  →  session / commands / types        （UI 层不许 import 具体后端）
render/  →  types / math3d                 （渲染层不许 import UI 层）
adapters/  →  types / math3d
```

`tests/test_layering.py` 是纯 AST 扫描，**必须能在没有 GPU 的机器上单独跑起来**。

## 坐标与朝向约定

- 矩阵**一律行主序**（`m[:3, 3]` 是平移）。只有 `math3d.to_gl()` 转置成列主序，
  **全项目只在上传处调用它**。
- 世界是 **Z-up**（MuJoCo 约定）。
- GL 纹理原点在**左下**：`ViewportImage.flip_y` 由后端申报，贴图时按它翻。

## 命令

```
make setup      # 建 venv、装依赖
make check      # lint + 全部单元/集成测试。提交前唯一需要记住的命令
make gpu        # 带 GPU 标记的用例
make reverse    # 把修复逐条去掉，判据必须立刻红（铁律 1）
make viewer     # 打开查看器（`make viewer SCENE=arm26`；`make assets` 列出全部）
make gizmo      # 打开一个带自由体的场景：手柄与 Ctrl+拖拽在那里才出现
make reflect    # 平面反射的展示场景
make gallery    # 每份场景渲一张图，只出图不判对错
make parity     # 与参照渲染器同机位对拍，出三联图
make calibrate  # 拿参照逐项标定光照与反射的混合律
make bench      # 逐 pass 量 CPU 与 GPU 耗时（`make bench SCENE` 换场景）
make showcase   # 一屏看全部已落地的渲染能力
make probe      # 重跑 GL 能力探测，刷新 docs/PLATFORM.md 的依据
```

Python 一律用 `.venv/bin/python`（`uv` 建的）。
