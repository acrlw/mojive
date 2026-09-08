# C++ 基础库选型与依赖边界

更新日期：2026-09-08。GLM 1.0.3 与 spdlog 1.17.0 已按固定提交纳入源码和构建；Eigen、EnTT 未加入。当前结果验证兼容性和生命周期，没有把依赖接入当成已测得的性能提升。

## 结论

**C++ 提供快速、稳定的舞台和基础设施，Python 定义业务与扩展。通用能力复用成熟库，Mojive 自己维护清晰的数据契约和所有权。** 依赖数量不是目标：少一个库，却多出一套自行维护的数学、日志或调度框架，通常不划算。引入库也应有当前用途，避免提前搬入完整游戏引擎的基础设施。

| 领域 | 建议 | 在 Mojive 中的范围 |
|---|---|---|
| 图形数学 | 首批采用 GLM | 相机、变换、四元数和几何计算，留在原生实现内部 |
| 数值线性代数 | 不纳入当前原生路线 | IK、最小二乘和业务求解由 Python 生态提供；不为这些业务加入 Eigen |
| 实体组件系统 | 暂不加入 EnTT | 原生编辑器需要大量可组合组件及查询时重新评估，不为增加对象数量而引入 |
| 原生日志 | 首批采用 spdlog | 承担 Mojive runtime 的日志输出与有界历史；优先使用编译库形式 |
| Python 日志 | 保留 Loguru，可选桥接 | 发布业务日志、配置与订阅；宿主程序继续自行选择日志库 |
| 渲染、窗口、UI、绑定 | 延续 bgfx、GLFW、ImGui、nanobind | 各自留在对应实现层，已有依赖管理方式继续使用 |
| 专用线程与同步 | 先用 C++20 标准库 | 明确所有权的仿真线程、取消、互斥和快照交接；复杂任务调度再选成熟库 |

GLM 与 Eigen 都可以处理小矩阵，选择 GLM 是当前图形基础设施的匹配判断，并非已经证明 GLM 比 Eigen 快。Eigen 不作为未来业务迁入 C++ 的预设依赖。EnTT 也不是被永久排除：如果需求演变到需要 ECS，优先使用成熟实现，不自行重写一个通用 ECS。

## 当前代码说明了什么

- `cpp/src/Contracts.cpp` 已使用 GLM 的右手系相机算法，替换验证阶段的手写向量和矩阵公式。输入退化检查、行主序边界及深度契约保留。
- `cpp/include/mojive/Render.hpp` 已有 `SceneSource`、`SceneFrame`、`objectId` 和连续的变换数组。标准库数据契约便于隔离后端，不要求内部也只能用标准库计算。
- Python 的 `Session` 已经负责选择、命令和编辑状态。增加另一份 ECS 世界，首先会产生状态归属、对象映射和同步问题，不能仅凭库的微基准判断收益。
- `python/src/mojive/log.py` 已用 Loguru；`ui/messages.py` 的 `OutputBuffer` 已是有界、带锁的日志历史。迁移期间它可以订阅原生记录；原生文件输出不以此 Python 面板为前提。
- 当前日志的 `configure()` 会调用无参数 `logger.remove()`，目前由 CLI 初始化调用。接入原生库时不能把它直接复用到模块导入或每个 Renderer 的构造过程，否则会移除宿主应用配置的 Loguru sink。

## GLM：使用现成算法，明确存储边界

GLM 提供图形向量、矩阵和四元数；名称中的 OpenGL 不意味着它依赖 OpenGL 运行库。官方手册也明确说明了按需包含头文件、列主序存储和类型对齐配置。[GLM 手册](https://github.com/g-truc/glm/blob/master/manual.md)

Mojive 的实现规则：

1. 内部直接使用需要的 GLM 类型和函数，不给每个加法、点积再包一层同义 API。自有辅助函数只表达坐标、相机、校验和数据转换等领域含义。
2. Python 和共享场景契约维持当前行主序、Z-up 约定。GLM 的矩阵存储与上传布局通过一个明确的边界转换衔接，不能直接把 NumPy 地址强转成 `glm::mat4*`。
3. 相机计算明确右手系和当前 `[-1, 1]` 深度契约，后端继续负责目标 API 的转换。不能因为换到 Metal 就全局改 GLM 宏，导致后端再次转换。
4. CPU 计算按功能选择 `float` / `double`；MuJoCo 状态保留其原精度，转换成 GPU 数据时才做相应精度收敛。库的默认类型不能决定物理精度。
5. 四元数分量顺序、矩阵 stride、对齐和编译宏集中规定。GPU 顶点布局继续显式声明，不依赖某个 GLM 类型碰巧具有的大小。
6. 按需包含头文件，避免所有源文件引入全部扩展。跨模块、NumPy 和资源边界保持明确数据格式；内部辅助头可以使用 GLM，不必为隔离而重复分配整帧数组。

IK、最小二乘、矩阵分解等业务能力由上层 Python 库承担，不放入 Mojive 原生功能路线。C++ 数学依赖只服务相机、几何和渲染等基础设施。Python 数值库自己的底层实现可以使用原生代码，不需要 Mojive 再封装一套同类求解 API。

## 行主序契约：来源与准确含义

当前实现的证据是：`math3d.compose()` 创建 NumPy 矩阵并将平移写入 `m[:3, 3]`；MuJoCo adapter 把 `xmat` 视作 3×3 矩阵；`math3d.to_gl()` 在 OpenGL 上传边界重新排列数据。原生 `Render.hpp` 延续行主序的共享契约，bgfx adapter 内的 `columnMajor()` 负责上传布局转换。

NumPy 支持 C-order、F-order 和带 stride 的视图；从普通嵌套序列创建数组时默认采用 C-order。MuJoCo 的矩阵采用行主序。因此当前选择与两者相容，也有利于保留现有 API。仅凭当前代码不能把最初设计动机归因于某一个库，而且两者都不要求 Mojive 内部永远使用同一存储布局。[NumPy 数组布局](https://numpy.org/doc/stable/reference/generated/numpy.array.html)、[MuJoCo 数据布局](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#data-layout)

契约必须把以下三个维度分别定义：

| 维度 | Mojive 约定 |
|---|---|
| 数学含义 | `pWorld = M @ pLocal`，齐次列向量；平移在 `M[:3, 3]` |
| 数据存储 | 共享的连续矩阵缓冲按行排列；NumPy 输入另外声明 shape、dtype、stride 和复制规则 |
| 坐标与投影 | Z-up、右手系及各个矩阵的空间含义；深度范围和图像朝向由契约与后端转换明确处理 |

行主序不意味着必须使用行向量，也不决定平移该放最后一行还是最后一列。`transform_points()` 使用批量点按行存储的 `points @ rotation.T + position`，与上述列向量数学约定等价。

Python 公开接口继续接收约定形状和含义的数组；接入原生层时按实际 stride 读取，在必要时统一规范化。默认快照提交保证数据所有权，不能把任意数组地址解释成 GLM 类型。若以后提供显式零拷贝接口，必须单独说明布局、可写性和有效期，不满足要求就明确报错。

内部 GLM 或 GPU 缓冲可以使用适合自己的布局；布局转换与已有的快照打包、GPU 上传合并，复用缓冲，避免来回分配与重排。改变内存顺序不会改变矩阵的数学含义，用户不需要因为后端从 OpenGL 换成 bgfx 就自行转置输入。

## EnTT：按编辑器需求引入

EnTT 的价值在于组件存储、组合查询和相关实体管理。它的 registry 不是开箱即用的全线程安全对象，组件修改与遍历仍有同步和引用有效性约束。[EnTT 官方 ECS 指南](https://github.com/skypjack/entt/wiki/Entity-Component-System#multithreading)

100 humanoid 这种场景，首先需要检查物理耗时、快照复制、批量变换、GPU 提交与读取，而不是把所有对象改为 ECS。现有平铺场景数据适合当前渲染入口；换存储系统是否更快，需要完整场景基准。

重新评估 EnTT 的触发点是：原生编辑器开始广泛组合 Transform、Camera、Light、Renderable 等组件，并需要按组件集合运行多种操作。若采用，registry 留在编辑器场景实现里；对外保留 Mojive 的 object ID，另建内部映射。MuJoCo body index、ECS entity 和用户可保存的 object ID 不互相冒充，渲染器继续消费场景快照。选择、撤销、远程协议和物理状态归属不会由 EnTT 自动解决。

## 日志：原生基础设施输出，Python 发布与订阅

**启用原生 runtime 时，Mojive 自己的日志输出与有界历史由 C++ 管理。** C++ 产生底层诊断，Python 产生业务记录并配置需要的输出端；Python 不是所有日志必经的转发线程。宿主程序的其他日志仍由宿主选择 Loguru、标准库 logging 或其他框架。

建议的最终路径：

```text
C++ 渲染 / 资源 / runtime 诊断 ─────────┐
                                      ├→ runtime 日志服务（spdlog）
Python 业务 → 原生发布接口 / 可选 sink ─┘     ├→ 原生有界 Output 历史
                                             ├→ 按配置异步写 stderr / 文件
                                             └→ 可选 Python 批量订阅
```

Python 可以直接使用 Mojive 的日志发布入口，也可以给自己的 Loguru / logging 添加一个只转发选定记录的适配器。不接管整个进程的 logger，不强制把宿主训练日志和其他库的消息全部送入 Mojive。原生代码始终能自行产生日志，不需要由 Python 提供内容。

### 输出与 GIL

原生线程不调用 Python sink。普通带 GIL 的 CPython 中，进入 Python 仍需取得 GIL；Loguru 的 `enqueue=True` 不能消除这一步。Python 发布业务日志时本来就在 Python 调用链里，跨边界复制必要字段后立即返回；后续文件写入和原生面板历史不依赖 Python 消费。[nanobind GIL 管理](https://nanobind.readthedocs.io/en/latest/api_core.html#gil-management)、[Loguru logger API](https://loguru.readthedocs.io/en/stable/api/logger.html)

spdlog 负责成熟的格式化、级别过滤、异步输出和文件管理；Mojive 增加小型结构化记录适配器与有界历史 sink，不另写通用日志框架。优先编译库形式，避免把整套模板实现带入每个源文件。[spdlog README](https://github.com/gabime/spdlog/blob/v1.x/README.md)

spdlog 异步队列默认在满时阻塞，因此诊断通道显式采用不等待队列腾位的策略，并统计丢弃量。服务生命周期内管理必要的日志 worker，不为每个窗口创建一个线程池。短锁、记录复制及格式化仍有成本，不承诺无锁或零开销。[spdlog 异步日志说明](https://github.com/gabime/spdlog/wiki/Asynchronous-logging)

### 明确的行为约定

- 输出端按需启用：库导入不自动写文件，不替换宿主日志配置。CLI 可显式配置 stderr / 文件；stdout 保留给调用方结果或机器可读输出。
- 记录包含发生时间、级别、component、runtime ID、来源线程、来源语言和序号，并拥有字符串数据。Python 异常可提交已格式化的 traceback；原生队列不保留异常对象或任意 Python 引用。
- Output 历史与异步队列有条数、字节及单条长度上限。发生丢弃时保留计数，不能无限增长，也不能为了等待 Python 而卡住原生生产者。
- 日志是诊断通道，任务失败走可靠的异常 / 任务结果。不会同时承诺固定内存、永不阻塞和任意洪峰下绝不丢日志。
- 正常热路径不逐物体、逐仿真步写文本；重复消息限频，性能数据使用计数和聚合指标。
- 订阅有独立读取位置、批量上限和 runtime 过滤；读取后先释放内部锁，再执行 Python 回调。慢订阅者不阻塞其他输出端。
- 文件由配置明确的一个端点写入，避免重复输出。桥接记录标明来源；同一条记录不得经 Loguru → native → Loguru 循环转发。
- logger、历史、sink、订阅与线程池由显式生命周期管理，不能使用全局清空操作影响其他 Mojive 实例或宿主应用。
- `close()` 停止提交，等待自有生产者完成，再排空原生日志并停止相关 worker；可选 Python 订阅在解释器仍有效时断开。异步 flush 请求本身不代表已经写完，正常关闭需要等待完成。硬崩溃时不保证尚未写出的队列记录。

### 现有 Python 版本如何迁移

当前 Python Output 面板暂时通过批量读取原生历史显示日志，Python 日志仍可用 Loguru。该面板是订阅者，原生文件输出不需要先经过它。原生 UI 迁入后直接读取同一类记录；无需为了日志消费而维持 Python UI 轮询。

仅在用户希望接入现有 Python 日志系统时，启用原生记录 → Python logger 的可选订阅，并避免该记录被再次转发回原生入口。未启用原生 runtime 的现有 Python 实现继续使用现有日志路径。上述迁移不要求普通 Python 脚本启动独立进程。

## MuJoCo 的复用范围

MuJoCo 自己也使用 tinyxml2、Qhull、tinyobjloader 等依赖；其完整性不来自拒绝第三方库。[MuJoCo 依赖构建文件](https://github.com/google-deepmind/mujoco/blob/main/cmake/MujocoDependencies.cmake)

Mojive 应直接复用 MuJoCo 的仿真、模型、动力学、已有数学工具和适用的并行能力，范围限于 MuJoCo adapter。它提供单步内部分计算的线程池支持，也支持只读模型配合各线程独立 `mjData` 的并行计算模式。[MuJoCo 多线程说明](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#multi-threading)

相机、通用场景和 renderer 不应因此强制依赖 MuJoCo：Mojive 还接收其他物理引擎、用户构造场景、远程发布和回放。MuJoCo 也不代替 Mojive 的选择、撤销、资源生命周期或 RHI。其内部数学实现不需要为统一库名而改写成 GLM / Eigen。

## 依赖管理与实际副作用

主要成本是编译时间、包体、类型布局、全局状态和升级兼容性。以下规则把成本限制在使用该库的模块中：

- 继续使用 `thirdParty/dependencies.json` 固定来源和提交，保留上游许可证；GLM / spdlog 首次接入时与对应实现一起加入 submodule，不先放一批未使用的依赖。
- 默认使用未经修改的上游代码；ImGui 的必要定制沿用已有独立修改记录。第三方 API 保持原命名，自有 C++ 继续 Qt 风格。
- 原生依赖尽量通过模块私有构建关系接入。共享契约不传递 `glm::mat4`、`entt::registry`、`spdlog::logger` 或 bgfx handle 到 Python 公开 API。
- 共享契约使用标准库不等于承诺跨编译器的稳定二进制 ABI。当前扩展与原生核心一起构建；将来若开放外部二进制插件，再设计版本化 ABI。
- spdlog 的格式化配置在 Mojive 自有构建中保持一致。初期使用它自带的 fmt 支持，若其他自有模块也确实需要独立 fmt，再通过上游提供的选项统一版本；不同时维护两套自有格式化依赖。[spdlog 构建选项](https://github.com/gabime/spdlog/blob/v1.x/CMakeLists.txt)
- 专用线程用标准库即可；C++20 不自带通用线程池。若开始需要任务依赖图、工作窃取或复杂调度，再选一个成熟实现，并统一线程预算，不自行扩建成调度框架。
- 各库升级独立提交，记录兼容性与包体变化。资源导入、压缩、几何处理等同样优先寻找已有能力或成熟库，但在对应功能迁移时选型。

## 下一步实施与验收

已在 C++ worktree 接入 GLM，并通过私有扩展提供 spdlog 日志发布、批量读取和关闭。原生 Output 面板与可选 Loguru sink 适配尚未接入。后续进度见[实现与验收清单](cpp-implementation.zh.md)。Eigen 与原生业务求解不在本轮路线内；EnTT 仅按基础设施需求重新评估，不改变当前 Python API。

数学验收覆盖退化相机、变换方向、投影和矩阵布局，并复用原生 RGB / 深度 / 分割及相机验收。日志验收覆盖无 GIL 的原生生产者、Python 发布与停止消费、原生输出持续工作、溢出计数、多 runtime 隔离、宿主 sink 保留、桥接无回路、显式关闭及解释器退出。性能比较继续使用 100 humanoid，记录正常日志级别下的帧耗时分位数、仿真吞吐、内存和包体；日志洪峰单独测试。

本文只更新方案和文档入口。本次未引入新运行时依赖、未重跑渲染性能基准，也不据此声称 Windows / Linux 已验证。当前阶段继续本机验证，不启动 workflow CI。
