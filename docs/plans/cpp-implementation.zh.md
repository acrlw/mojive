# C++ 实现与验收清单

更新日期：2026-09-08。开发分支：`codex/cpp-foundation`。

**已有 Python API 和行为是兼容基准。第一阶段已经打通 Python → C++ → bgfx → NumPy 的离屏渲染；完整后端迁移尚未完成。** 当前公开 Renderer 和 Viewer 继续使用现有实现，Python 安装流程不变。

## 已完成的第一阶段

| 内容 | 实现与验证 |
|---|---|
| 依赖管理 | GLM 1.0.3、spdlog 1.17.0 固定提交；9 个核心 submodule；ImGui 281 个原始文件基线完整 |
| 图形数学 | GLM 替换手写相机公式；行主序、齐次列向量、Z-up、右手系和 `[-1,1]` 投影契约保留；随机相机与原 Python 结果比较 |
| 原生日志 | 有界历史、独立订阅游标、发生时间与来源线程、UTF-8 截断、丢弃计数；可选异步文件输出；关闭等待原生输出线程退出 |
| 私有绑定 | nanobind 构建 `mojive._native`，Python 方法保留 snake_case；未接管宿主 Loguru 配置，与 MuJoCo 两种导入顺序共存 |
| 线程归属 | `RenderRuntime` 在自有线程创建、调用、销毁后端；队列最多等待 64 项；接受的调用在关闭前完成；异常传回调用者 |
| 数据与结果 | 场景和帧提交取得自有数据，帧暂存按调用线程复用；输出数组拥有存储；支持 RGB、uint32 ID、int32 分割对、float32 深度和帧来源信息 |
| 实际场景 | Python 提交官方 100 humanoid 场景的 5,101 个渲染实例，连续 12 帧；检查输出 ID 合法性并保存画面 |
| 生命周期 | 多个调用线程、同一场景多个目标、关闭目标互不影响、过期帧取消、初始化失败后重试、重复关闭和解释器自动释放 |

上述测试在 macOS / Metal 完成。它们验证行为和边界，不构成相对现有 Python 渲染器的性能提升结论，也不代表 Windows / Linux 已验证。

## 当前私有接口的明确语义

- `set_scene()` 和 `update()` 返回时，后端已经消费本次 CPU 数据；不代表画面已完成 GPU 执行。调用期间输入必须保持稳定，调用后允许修改或释放。
- Python 浮点输入按形状转为连续 float32，支持 F-order 和带 stride 的视图。ID 与索引输入要求相应的连续 uint32 / int32，避免隐式整数转换破坏身份信息。所有矩阵仍按数学上的行、列解释，用户不转置。
- `render()` 返回帧 token。`readback()` 返回有界读取 ticket；`advance()` 推进提交，`poll()` 检查状态，`wait()` 推进并等待结果。当前不是 `asyncio.Future`。
- `ReadbackResult.image` 在 Ready 时创建独立、连续、左上原点的 NumPy 数组；Pending / Canceled 时返回 `None`。已有输出在运行时关闭后仍有效。复用 `out` 数组尚待接入公开外观层。
- `close()` 拒绝后续工作，等待已接受调用完成，在原生所属线程释放 GPU，然后停止日志输出。正在同步等待读取的调用有超时；关闭不是强制中断驱动程序。
- 队列满时对提交方施加等待，等待时释放 GIL；日志满时按约定丢弃并计数。两个通道的策略不同。
- 原生 worker 不调用 Python。绑定在持有 GIL 时取得输入快照，随后释放 GIL 等待原生工作。物理状态的稳定性依旧由其现有所有者保证。
- 当前一份 bgfx runtime 只拥有一个场景，可创建多个相机/输出目标。**这不等价于已经支持多个独立公开 Renderer。** 相互独立的场景和资源必须在下一阶段解决，不能通过每帧切换时销毁重建 GPU 场景应付。
- 离屏 runtime 不接受平台窗口。窗口事件、原生句柄创建与 ImGui 主线程约束将在 Viewer 接入时单独处理。

## 后续验收顺序

| 阶段 | 必须完成的内容 | 状态 |
|---|---|---|
| 2. Python 兼容外观层与资源隔离 | 现有 SceneSource / SceneFrame 的适配；共享设备下的独立场景；多 Renderer 交错更新和关闭；现有同步、读取 ticket、`out` / stride / dtype / 异常语义 | 待实现 |
| 3. 渲染效果与性能 | 材质、纹理、透明、灯光、阴影、反射、动态网格、调试和选择输出；按同画质比较现有后端；100 humanoid、复杂网格、反复加载与内存增长 | 待实现 |
| 4. Viewer 和 UI 基础设施 | 主线程窗口/ImGui、原生渲染交接、现有 Session 权威状态；面板扩展边界；原生 Output 订阅；曲率连续自绘控件与统一设计规范 | 待实现 |
| 5. 分发与平台验收 | 可安装原生 wheel、shader / 动态库打包、独立环境安装、Python 版本矩阵、Windows/macOS/Linux 生命周期与 GPU 验收 | 待实现 |

公开后端注册以兼容性和能力覆盖为前提；不把尚未实现的效果静默忽略。普通脚本继续直接调用 Python，不要求先启动 daemon。独立 runtime CLI、远程 session 和崩溃隔离留作之后的产品能力，不混入当前渲染迁移。

C++ 不承接 IK、最小二乘、规划等业务求解，不引入 Qt、Eigen 或 EnTT。GLM、spdlog、bgfx 和 nanobind 保持实现边界，后端无关的头文件不暴露它们的类型。当前没有开启 workflow CI。

## 本地复现

```bash
make check
make cpp-test
make cpp-python-test
make native-bindings-test
make native-fixture HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
make cpp-python-gpu
make cpp-probe
make cpp-gallery HUMANOIDS_MODEL=/path/to/mujoco/model/humanoid/100_humanoids.xml
```

共享测试要求见[验证矩阵](../guides/testing.md)。原生运行时的线程、异常和关闭还可用 `MOJIVE_ENABLE_SANITIZERS=ON` 在独立构建目录运行 CTest；GPU 检查顺序执行。

当前私有扩展有 6 项 CPU/Python 检查及 4 项 GPU 检查；C++ CTest 有 3 项；原有绑定对比有 13 项。现有 Python 快速测试 1,625 项、集成测试 129 项通过。生成结果保存在 `output/cpp-python/bgfx/`，其中 `products.png` 用于查看三个方向与颜色位置，`humanoids100.png` 和 `acceptance.json` 记录实际大场景输出。
