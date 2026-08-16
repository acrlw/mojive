# 本机实测的 GL 事实

规格（`prompt/02-tech-stack.md`）里的数字来自 NVIDIA RTX 5090 / GL 4.3。本项目的开发机不是
那台。**下面每一条都是在本机跑出来的，不是推断**——探测脚本见 `tools/probe_gl.py`，
`make probe` 可复跑。

平台：macOS 26.6.1 / Apple M5 / GL **4.1 core（Metal 后端）** / GLSL 410。

| 项 | 实测值 | 对设计的影响 |
|---|---|---|
| `GL_VERSION` | `4.1 Metal - 90.5` | 规格的 3.3 core 下限**正好合适**：macOS 封顶就是 4.1，没有 SSBO、没有计算着色器、没有 `BaseInstance`。规格为下限所做的每一条让步在这里都是必需的，不是保守 |
| `GL_MAX_SAMPLES` | 8 | 4× MSAA 可用 |
| `GL_MAX_VERTEX_ATTRIBS` | 16 | 3 个网格属性 + 7 个实例属性 = 10，留 6 个余量 |
| `GL_MAX_TEXTURE_IMAGE_UNITS` | 16 | 够用（贴图 + 阴影图集 + skybox + id） |
| **`R32UI` 且 `samples > 1`** | **`INCOMPLETE_ATTACHMENT`** | **本机不支持多重采样整数附件**（纹理与 renderbuffer 都不行，2/4/8 档全不行）。见下方 §1 |
| `RGBA8 ms4 + depth ms4` | OK | 主目标的 MSAA 照常 |
| `RGBA8 + R32UI + depth`（非 MSAA） | OK | 回退路径可行 |
| `Framebuffer.clear()` 打整数附件 | 写进浮点位模式（`0.9f` → `1063675494`） | 规格 §2.4 在本机**如实复现**。必须换清法 |
| 全屏三角形清整数附件 | 读回**精确 0** | 采用它：纯 moderngl、可移植、无 ctypes |
| `glClearBufferuiv`（ctypes） | 可用，`glGetError() == 0` | 作为快路径保留，不作为唯一实现 |
| **深度写掩码重放** | **复现**：上一帧 `depth_mask=False` → 这一帧 `clear` 不清深度 → 同批几何一个片元都不写 | 规格 §2.4 的坑是真的。清屏前显式开深度写之后恢复正常 |
| GPU 计时查询 | **可用**：`GL_TIME_ELAPSED` 包 50 次绘制 = 22.77 ms | `timestamp` 档不试，`elapsed` 档够用 |
| `GL_KHR_debug` | **缺失**（`GL_ARB_debug_output` 也缺失） | `glPushDebugGroup` 必须退化成空操作。规格 §2.6 已经为此留了口子 |
| **`GL_LINE_WIDTH_RANGE`** | **`[1.0, 1.0]`** | 规格 §2.2 第 4 条在本机**逐字成立**。所有粗线只能展开成三角形带，没有第二条路 |
| 几何着色器 | OK | `WIREFRAME` 的重心坐标路径可行 |
| `usampler2DMS` + `texelFetch` | 编译通过 | 留给支持整数 MSAA 的机器 |
| `floatBitsToUint` | OK | GLSL 330 起可用 |
| `glPolygonMode`（`ctx.wireframe`） | OK | 调用方可能留下它——`GLStateGuard` 必须守 |
| HiDPI | 窗口 320×240 → 帧缓冲 640×480，`content_scale = 2.0` | 规格 §10.4 不是理论问题，本机就是 2× |
| ctypes 实例属性字节偏移绑定 | **可用**，`glGetError() == 0` | 规格 §4.2 的做法能落地，见 §2 |

## 1. ID buffer 不能和 MSAA 主目标同住（本机）

规格 §7.1 假定 ID buffer 是 `opaque` 的第二个颜色附件，跟主目标同样 4× MSAA，取值靠
`texelFetch(usampler2DMS, coord, 0)`。**本机做不到**：任何采样数下 `R32UI` 附件都让
FBO 不完整。

不硬编码任何一种，**建目标时探一次**，结果落进 `caps.id_msaa`：

| 布局 | 条件 | 形态 |
|---|---|---|
| `SHARED` | 整数 MSAA 附件可用（NVIDIA/桌面） | 规格原样：一个 FBO 两个颜色附件，`opaque` 一趟写两份 |
| `SPLIT` | 本机 | 主 FBO 只有 `RGBA8 ms + depth ms`；另有一个 1× 的 `R32UI + depth` 目标，`opaque` 之后跟一趟极廉价的 id 子 pass（只有位置与 id，无光照无贴图） |

下游（拾取、描边、分割图）一律通过 `RenderTarget.id_texture` / `.id_samples` 取值，
着色器用 `#define ID_MULTISAMPLE` 切换 `usampler2DMS` / `usampler2D`——**两条路径的取值
语义完全一样**，差别只在采样器类型。

代价诚实记下：`SPLIT` 下不透明几何要走两遍顶点。id 子 pass 没有光照、没有贴图、没有阴影
采样，实测量级与 shadow pass 相当。

## 2. 实例属性的字节偏移：moderngl 没有这个入口

规格 §4.2 要求"每桶一个 VAO，实例属性缓冲用字节偏移重新绑定"。moderngl 5.12 的
`Buffer.bind(*attribs, layout=None)` **没有 offset 参数**，格式串里的 `x` 填充会同时改变
stride，凑不出来。

做法：VAO 由 moderngl 正常建，建完之后用一段 ctypes 把实例属性的指针**重新指到桶的基地址
上**（`glVertexAttribPointer` + `glVertexAttribDivisor`）。这只在 `set_scene` 跑，每帧
零开销，之后 `vao.render(instances=n)` 照常。实测五个实例三档基址，颜色逐档正确、
`glGetError() == 0`。

拿不到原生 GL 符号的平台上退回 **每桶一个缓冲**：所有不变量都还在（桶只算一次、热路径不
分配、绘制次数不变），差别只是"一次上传整段"变成"每桶一次上传"，B 通常是 6～20，
代价是几十微秒。两条路径由 `InstanceStore` 一个接口盖住，同一份用例对两条都跑。

## 3. 逐 pass GPU 计时在 TBDR 上会把工作算到别人头上

`GL_TIME_ELAPSED` 在本机**可用**（探测到 `elapsed` 档），但**逐 pass 的数字不能直接信**。

实测（`many_objects`，1904×1850，601 实例 / 21.2 万三角形）：

| pass | CPU ms | GPU ms（照常跑） | GPU ms（每条 pass 结束前强制 `finish`） |
|---|---|---|---|
| `opaque` | 1.07 | 0.94 | 0.80 |
| `id` | 0.53 | **0.00** | **0.26** |
| `present` | 0.03 | **0.00** | 0.00 |

原因：Apple 的 GPU 是**分块延迟渲染（TBDR）**，绘制命令被攒到 tile flush 才真正执行，
**谁的查询还开着就算给谁**。于是后面几条 pass 的时间被并进了 `opaque`。

不为此加 `finish()`：那正是 04 §4.7 / 13 §13.4 明令禁止的"每帧让 CPU 等一次 GPU"——
**为了把账算清楚而付上真实的帧时间，是本末倒置**。

处理：
- 照实报驱动给的数字，**不拿 0.0 顶替"没测到"**（那看起来像"GPU 快到不要钱"）；
- `make bench` 在发现"有的 pass 为 0、有的不为 0"时**打一条提示**指向本节；
- **整帧的 GPU 合计仍然是可信的**（tile flush 总会发生在某条查询里），逐条的分摊不可信。

换到立即模式的 GPU（NVIDIA/AMD 桌面）上逐条就准了，规格 13 §13.1 那张分项表正是在
那种机器上量的。

## 4. shadow pass 在本机是**填充率受限**，和规格的机器反过来

规格 06 §6.7 在 RTX 5090 上量到 shadow pass **GPU 0.018 ms / CPU 0.117 ms**，并据此说
"CPU 是 GPU 的 6.5 倍"。**本机整个反过来**（1280×720，4096² 图集，3 级）：

| | 本机 | 规格 |
|---|---|---|
| shadow GPU | **0.53 ms** | 0.018 ms |
| shadow CPU（扣掉帧边界等待与计时查询） | **0.081 ms** | 0.117 ms |
| 其中级联矩阵计算 | **0.051 ms** | 0.052 ms |

拆开量过：只清屏 GPU **0.0000 ms**（分块 GPU 的 fast clear）；画 1/2/3 级 =
0.255/0.352/0.450 ms；3 级但只画第一个桶（占 83% 三角形）只要 0.169 ms——
**差额几乎全在那一片铺满视口的地面上，是填充率不是三角形数**（三块 2048² 瓦片 =
12.6 M 片元）。

**结论对优化取向的影响**：规格 §6.7 说"级联矩阵计算 0.052 ms，批量化能收掉，但那是带
纹素吸附的矩阵数学，改动风险高于收益"——**在本机这条更成立**：0.58 ms 的 pass 里去收
0.05 ms，一分风险都不该冒。没有动它。

（级联矩阵那 0.051 ms 与规格逐值吻合，因为它两边都是三次逐级的小 numpy 调用，
全是 per-call 开销，与 GPU 无关。）

## 5. 整数顶点属性：本机分辨不出对错

`object_id` 是 `uint32` 实例属性，按 GL 规范必须用 **`glVertexAttribIPointer`** 指定；
用 `glVertexAttribPointer` 指定一个整数着色器输入，结果是**未定义的**。

**本机的驱动按位重解释，两者读回来逐位相同**（实测：写 `16777217`，两条路径都读回
`16777217`，`glGetError` 都是 0）。

后果要写清楚：

- 代码仍然走 `glVertexAttribIPointer`——**规范说了未定义就是未定义**，别的驱动完全可以
  按定点数转换，那时 2²⁴ 以上的 id 会悄悄丢精度，而**画面照常，只有拾取和描边错位**；
- 但**这条不能用画面判据来守**：在这台机器上它永远红不了，写进反向验证只会是一条假绿
  （12 §12.1 三种假绿之外的第四种：**判据落在了一个本机分辨不出的量上**）。
- 能守住的是**声明的类型**（`INSTANCE_ATTRIBUTES` 里那一项必须是整数类型），
  反向验证就落在那里。换到会区分的驱动上再补画面判据。

## 6. `mujoco-classic` 在 macOS 上的地位（以及它**不**影响什么）

`mjr_` 是 GL 1.5 固定管线，要 compat/legacy 上下文；forge 要 core profile。
规格 §1.4 说这两条路径只能**进程级**二选一——所以 `mujoco-classic`
（在查看器进程里用 `mjr_` 当渲染后端）在 macOS 上起不来。
`backends` 命令**如实报**这一条，而不是列出来再让用户撞墙。

**但这不影响 `make parity`。** 曾经把这两件事混为一谈，结论下成"macOS 上跑不了对拍"，
白白搁置了整个对拍关卡。实际上：

- GL 1.5 的固定管线是 **GL 2.1 的子集**，macOS 的 legacy 上下文跑得动 `mjr_`；
- `mujoco.Renderer` 在本机**离屏出图实测正常**（真图，有阴影有反射）；
- 规格 §12.5 本来就写了参照渲染器跑在**子进程**里——子进程建自己的上下文，没有冲突。

不能共存的是**上下文**，不是**机器**。`make parity` / `make calibrate` 现在都跑得了，
而且一跑就找出了三个真缺陷（见 `docs/RENDERER.md` 的"已推翻"一节）。

## 7. moderngl 没有 stencil（影响平面反射的做法）

**实测**：`moderngl 5.12.0` 的 `Context.enable()` 能收的只有
`BLEND / CULL_FACE / DEPTH_TEST / PROGRAM_POINT_SIZE / RASTERIZER_DISCARD` 那几位，
**没有 `STENCIL_TEST`**；`Context.framebuffer()` 的深度附件只能是
`depth_texture()` / `depth_renderbuffer()`，两者都是**纯深度**，
建不出 `DEPTH24_STENCIL8`。

```python
>>> [a for a in dir(moderngl) if a.isupper() and "STENCIL" in a]
[]
```

**影响**：规格 14 §M5.2 的平面反射写的是"stencil 限制在平面轮廓内"。那条路要走的话
得整条自己用 ctypes 补：附件格式、`glStencilFunc`、`glStencilOp`、清 stencil。
本项目改用"渲进一张离屏反射图、反射面按屏幕坐标采它"，同样限制在平面轮廓内
（只有 `reflectance > 0` 的片元会去采），代价也一样是把不透明几何再渲一遍。
取舍与理由记在 `DECISIONS.md §一.16`。

**用到的是 GL 3.3 core 本来就有的用户裁剪面**（`gl_ClipDistance[0]` +
`glEnable(GL_CLIP_DISTANCE0)`，core 保证至少 8 个），不需要扩展；
moderngl 没有这一位的枚举，走 `Context.enable_direct(0x3000)` 把裸 enum 递进去。
