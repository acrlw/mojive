# forge-viewer

可独立使用、也可接物理引擎的 3D 场景查看器，核心是自研的现代 OpenGL 渲染器 **forge**。

打开一个模型文件（MJCF `.xml` / URDF），在一个窗口里：看它、转视角、点选零件、
用鼠标推它拧它（物理有反应）、暂停之后用手柄精确摆放、拖关节滑条看曲线切渲染开关、
从脚本往画面里画三维标注。

使用者是**做机器人仿真与强化学习的工程师**，不是美术，不是游戏玩家。

## 为什么自研渲染器

现有仿真查看器有同一个毛病：**渲染器是仿真库的附属品，十几年没动过**。MuJoCo 的 `mjr_`
是 OpenGL 1.5 固定管线，没有一行着色器，几百个几何体就是几百次绘制调用，没有性能出口。

而调试机器人时，画面质量**不是审美问题，是信息问题**：

| 缺什么 | 后果 |
|---|---|
| 看不出零件边界 | 分不清是哪一节连杆卡住了 |
| 没有阴影 | 判断不出物体离地面多远 |
| 高光过曝成死白 | 看不出表面朝向 |
| 没有 X 光轮廓 | 选中一个被遮挡的零件之后不知道它在哪儿 |
| 没有逐 pass 计时 | 卡了不知道卡在哪 |

**每一个渲染特性都必须能回答"它帮到了哪个调试场景"。** 答不上来的不做。

## 上手

```bash
make setup                      # 建 venv、装依赖（uv + Python 3.11）
forge-viewer view test_scene    # 打开默认场景
forge-viewer backends           # 本机能跑哪几条路径，缺什么会说清楚
forge-viewer assets             # 列出内置场景
```

## 命令

```
forge-viewer view    <asset> [-b BACKEND] [--paused] [--no-vsync]
forge-viewer canvas  [--demo canvas|lighting|text]
forge-viewer toy     [--no-vsync]
forge-viewer conformance [BACKEND] [--asset ASSET]
forge-viewer serve   <asset> [--host HOST] [--port PORT]
forge-viewer attach  [--host HOST] [--port PORT] [--debug-view VIEW]
forge-viewer replay  <snapshot> [--loop] [--speed FACTOR]
forge-viewer doctor  <asset>                 # 跑 90 帧自检，坏了退出码 1
forge-viewer inspect <asset> [--json]        # 只打印场景树与关节表，不开窗口
forge-viewer capture <asset> -o out.png
forge-viewer record  <asset> -o out.mp4 [--frames N] [--fps FPS]
forge-viewer audit   <asset> [--json] [--strict]
forge-viewer backends / assets / probe
```

`--json` 时 **stdout 上只有那一份文档**，引擎日志走 stderr。

## 门禁

| 命令 | 做什么 |
|---|---|
| `make check` | lint + 全部单元/集成测试。**提交前唯一需要记住的命令** |
| `make golden` | 基准图回归（只比对）；写基准要显式 `make golden-accept`，**因为重新生成之后必须肉眼看一遍** |
| `make gpu` | 需要 GL 上下文的用例，**一个文件一个进程** |
| `make reverse` | 把修复逐条去掉，判据必须立刻红（见下） |
| `make gallery` | 每份场景渲一张图。**只出图，不判对错** |
| `make bench` | 逐 pass 量 CPU 与 GPU 耗时，取中位数 |
| `make showcase` | 一屏看全部已落地的渲染能力 |
| `make parity` | 与参照渲染器同机位对拍，出三联图（参照跑在子进程里） |
| `make calibrate` | 拿参照渲染器逐项标定光照。**改任何颜色系数之前先跑它** |
| `make probe` | GL 能力探测，`docs/PLATFORM.md` 的依据 |
| `make canvas` | 不加载 MuJoCo，打开可用 Python 增删和移动物体的 3D 画布 |
| `make toy-physics` | 正式非 MuJoCo 物理 adapter：重力、地面碰撞、播放/step/Reset 和位姿编辑 |
| `make adapter-conformance` | 无窗口检查实例列、节点图、mesh、逐帧位姿、动态网格与 timestep 契约 |
| `make lighting` | 无物理场景的 spot / point / area 灯光；Hierarchy 选灯后可在 Inspector 编辑 |
| `make capture` | 截图到项目的 `output/capture.png`；可用 `SCREENSHOT=...` 覆盖 |
| `make text-overlay` | 验收世界空间文字的锚点、屏幕偏移/对齐和 depth/always 遮挡 |
| `make gizmo` | 原生位姿手柄验收：默认 2D、F9 可切 3D，G/R 平移旋转、T 切 body/world frame |
| `make perturb` | 物理扰动验收：Ctrl+左拖平移，Ctrl+右拖扭转；扭转标记只画二维实线剪影 |
| `make robot` | 按需下载并打开官方 MuJoCo Menagerie 模型；默认 Unitree Go2，可用 `ROBOT=unitree_g1` / `unitree_h1` 切换 |
| `make mujoco-audit` | 无窗口审计模型中会被支持、隐藏、降级或跳过的 MuJoCo 可视化能力 |
| `make mujoco-visuals` | 交互验收 heightfield、site、tendon、contact point/force |
| `make cameras` | F6 在自由相机与 MJCF named camera 间切换，动态相机随 body 更新 |
| `make geom-groups` | F9 切换 MuJoCo visual group 0–5（geom / site / flex / skin）；画面和物理射线使用同一组 geom 掩码 |
| `make deformables` | 验收 flex 1D/2D/3D 与骨骼 skin；动态顶点直接更新原 GPU mesh |
| `make record` | 把视口流式编码为 MP4，不在内存中积攒整段视频 |
| `make pvd` | 一个物理进程发布快照，打开相互独立的效果窗口和 normal debug 窗口 |
| `make snapshot-record` | 把结构、物理帧和 debug commands 录成 `.fvs`；Ctrl-C 结束 |
| `make snapshot-replay` | 不启动物理，循环回放 `.fvs` 并打开正常 viewer |

`make capture SCENE=deformables` 会在终端打印绝对路径，默认为
`output/capture.png`；`make record` 默认写 `output/recording.mp4`。

### 两条判据纪律

**一、判据必须在移除修复时变红。** 不红的判据看着是绿的，实际什么都没守住。
`make reverse` 就是把这件事做成可重跑的——它逐条去掉一处修复，确认对应的判据确实变红。

**二、渲染类判据必须走窗口路径。** `capture()` 在 numpy 里翻了一次 Y；曾经整个画面在窗口里
上下颠倒、60 多条测试全绿、两天没人发现，因为所有渲染判据都坐在那两条自相抵消的离屏路径上。

## 落到哪一步了（对 14 · 实现顺序）

| 步 | 内容 | 状态 |
|---|---|---|
| **M0** 骨架 | 窗口 + GL 上下文 + imgui 停靠 + `Session`/`SceneAdapter`/`RenderBackend` + 命令/查询 | ✅ |
| **M1** forge 第一帧 | 上下文挂接 + `GLStateGuard` + 渲染目标 + opaque + present | ✅ 三条判据都立住了 |
| **M2** 材质与纹理 | SoA `RenderScene`、分桶实例化、Blinn-Phong、颜色空间与色调映射、贴图、透明桶 | ✅ 已与参照对拍，见下 |
| **M3** debug draw | 7 图元、层与遮挡档、屏幕空间尺寸、外部 socket | ✅ 一万条线 0.33 ms/帧、1 次绘制 |
| **M4** 调试能力 | ID buffer、拾取、选中轮廓、调试视图、逐 pass 计时、Stats | ✅ 逐 pass GPU 计时在本机有平台限制，见 `docs/PLATFORM.md` §3 |
| **M5.1** 方向光 CSM | 3 级级联 + 4096² 图集、bias 落采样端按斜度、纹素吸附 | ✅ 掠射高频能量 0.0148（判据 < 0.05）|
| **M5.2** 平面反射 | 镜像相机 + 斜裁剪面 + winding 翻转 + 离屏反射图 | ✅ |
| **M5.3–5.5** 本地投影光 | spot / point 六面阴影、8 个本地阴影槽、16 灯光照、`light_range` 裁剪 | ✅ 8 灯与单灯反向判据已验 |
| **M5.6–5.7** 面光 / 雾霾 | area 软阴影近似；fog 与 haze 独立、在线性光阶段合成 | ✅ 开关与退回硬阴影均可逆 |
| **M6** 输出与相机 | 任意分辨率截图、流式录像、正交相机无跳变切换 | ↪ 输出/相机已完成；原计划的 ImGui platform multi-viewport 已取消，PVD 改用独立进程窗口 |
| **M7** 第二条物理后端 | 通用 adapter 接 forge | ✅ static 工具后端 + 正式 `ToyPhysicsAdapter`；两者都不 import MuJoCo |

## 不带物理后端使用

`Scene` 直接产出渲染层需要的结构与逐帧位姿，不 import MuJoCo。增删物体会增加
`structure_revision`，只移动物体不会重建场景：

```python
from forge_viewer import Scene, build_scene

scene = Scene()
ball = scene.sphere(name="ball", position=(0, 0, 0.5))
viewer = build_scene(scene)

for frame in range(300):
    ball.set_pose((frame * 0.01, 0, 0.5))
    viewer.sync()
viewer.release()
```

显式灯光也是 Forge 场景实体，不是 MuJoCo adapter 的附属数据。它们会出现在 Hierarchy，
Inspector 可编辑类型、颜色、强度、局部位置/方向、衰减、范围、阴影与 area 半径；自定义物理
后端只需在 `SceneFrame.lights` 提供 body-attached light 的动态世界位姿。

接自己的物理引擎时继承 `SceneAdapterBase`，最小实现只有 `scene_source()`、`frame()` 和
`step()`；关节、驱动、射线、扰动等能力都有可选的默认空实现。用
`build_from_adapter(adapter)` 进入同一套窗口、拾取、描边、debug draw、阴影和录制链路。
`AdapterCaps.simulation` 明确区分会步进的物理世界与静态工具，不再用“是否恰好实现了某个
MuJoCo 方法”来猜。`ToyPhysicsAdapter` 是仓库内正式的第二物理后端：有自己的 timestep、
重力与地面碰撞，刻意不实现关节、raycast 和 perturbation，用来验证 UI 对缺失可选能力的降级。
`make toy-physics` 肉眼验收，`make adapter-conformance` 验收公共契约；MuJoCo 也跑同一套检查：
`make adapter-conformance ADAPTER=mujoco CONFORMANCE_ASSET=deformables`。

## PVD 式双窗口

`make pvd` 不启用 ImGui platform multi-viewport。它启动一个无窗口物理 publisher，效果与
debug 各是一个完整、独立的 viewer 进程，因此布局、相机和窗口生命周期互不干扰。结构变化
可靠送达，普通帧只保留最新一份；即使渲染卡顿或窗口被拖住，也不会反压物理步进。暂停、step、
qpos/ctrl、位姿与 perturbation 通过独立命令通道返回明确结果。

也可以拆开运行：先 `make serve PVD_SCENE=deformables`，再开任意数量的
`make attach ARGS="--debug-view normal --title debug"`。该协议目前用于本机可信进程，传输数据
使用 Python pickle，不应直接暴露到不可信网络。

attach 窗口各自使用自由相机；远程协议暂不逐帧传输挂在运动 body 上的 named camera。
本地 `make cameras` 的 free / named camera 切换不受这个边界影响。

`make snapshot-record PVD_SCENE=gizmo SNAPSHOT=out/bug.fvs` 会记录与远程 viewer 完全相同的
packet；随后 `make snapshot-replay SNAPSHOT=out/bug.fvs` 可在物理进程已经退出后复现场景。
回放是只读的，避免 UI 看起来能改参数、实际却没有物理世界接收。

adapter 作者需要的 `SceneSource` / `SceneFrame` / `FrameNeeds` / `MeshUpdate`，以及工具侧常用的
`DebugDraw` / `Occlusion` / `RenderFlag` / `CameraView` 都从 `forge_viewer` 顶层导出；批量
`points()`、`lines()`、`arrows()` 可直接服务本地脚本、CLI 或后续 RPC，不依赖 MuJoCo。
`ToyPhysicsAdapter` / `check_adapter()` / `SnapshotPublisher` / `SnapshotWriter` 也都是公开接口，
第三方引擎不需要 import 项目内部路径。

### 对拍跑起来之后

`make parity` 与 `make calibrate` 都能跑（参照渲染器在**子进程**里，规格 §12.5 的原话）。
一跑就找出三个真缺陷，全部已修：

1. **环境光系数**：规格说 ×2，MuJoCo 3.11.0 上参照**只算一遍**（0.5 → 121，不是 241）。
   钉它的判据当时是**自证**的（期望值与被测代码用同一个常量），所以一直没发现。
2. **逐灯 `ambient` 整个没接**：参照对头灯与逐灯 ambient 是**相加**的。
3. **贴图上传时被压坏**：CPU 线性化后存回 8 位，原始 40 只剩 5（量化误差 18.5%）。
   改成用 `GL_SRGB8_ALPHA8` 让硬件解码。

对拍指标本机实测 IoU 0.25 / 块差 17.9，比规格的 0.64～0.80 / 10～13 差——
**原因已定位并量化**：`mjr_` 的全部光照运算在**显示域**做，forge 在**线性域**做
（05 §5.4 的要求）。单个乘积几乎恰好往返，**求和不行**，地板亮度比恒为 0.774，
模型与实测逐值吻合。这是一处**"我们更好"**，已登记进 `docs/RENDERER.md` 并接受数字变差。

## 结构

```
src/forge_viewer/
├─ types.py math3d.py commands.py   三方共同的词汇表（不依赖任何一层）
├─ session.py                        唯一真源
├─ scene.py                          无物理程序化场景（稳定 object id + 动态结构）
├─ adapters/                         场景来源（MuJoCo / static / 自定义物理）
├─ render/
│   ├─ scene.py                      RenderScene（SoA）
│   ├─ backend.py                    RenderBackend 协议
│   └─ forge/                        自研渲染器
│       ├─ passes/                   shadow → reflect → opaque → id → skybox
│       │                            → transparent → outline → debug → gizmo → present
│       └─ shaders/
└─ ui/                               窗口、面板、交互
```

分层由 `tests/test_layering.py` 守：**渲染层不许 import UI 层，UI 层不许 import 具体后端，
渲染层不许 import 任何物理库**。它是纯 AST 扫描，**在没有 GPU 的机器上也能单独跑起来**——
否则一旦 pytest 因为别的用例需要 GPU 而整体跑不起来，这条约束就形同虚设。

## 文档

- [`docs/PLATFORM.md`](docs/PLATFORM.md) —— 本机实测的 GL 事实。规格里的数字来自另一台
  机器，**冲突时以它为准**。`make probe` 可复跑。
- [`docs/RENDERER.md`](docs/RENDERER.md) —— 与参照渲染器的**差异登记表**。每一处差异只能
  落进"我们错了"（修）或"我们更好"（登记，接受对拍数字变差）两类，**不允许含混**。
- [`docs/ROADMAP.md`](docs/ROADMAP.md) —— MuJoCo 剩余覆盖、PVD 式远程双窗口与第二物理后端
  的分阶段计划；每阶段都先定义 `make` 验收入口。
- [`docs/DECISIONS.md`](docs/DECISIONS.md) —— 与**规格文本**不一致的地方：照字面做会错的、
  本机做不到的、**已经拿对拍裁决的**，以及**规格自己那几条假绿判据**。
  量出来但没有改的也在这里——不假装没看见。
- 完整规格：`../prompt/`（01～14）。

## 约定

- 矩阵**一律行主序**（`m[:3, 3]` 是平移），只有上传处转置成列主序。
- 世界是 **Z-up**。
- 注释与文档用中文，解释**为什么**而不是**是什么**。
