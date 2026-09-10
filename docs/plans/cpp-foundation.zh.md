# C++ 正式开发前的准备

**产品仍然是从 Python 使用的 Mojive。** 保留 `mojive.Renderer` 类似 `mujoco.Renderer` 的用法，以及现有 Viewer、SceneRenderer、NumPy 输出和上下文管理方式。C++ 是实现层，负责后续迁入的渲染、批量数据处理、原生任务调度和资源生命周期。

## 原生层的职责边界

C++ 提供舞台和基础设施，Python 负责上层业务、算法选择、工作流与二次开发。IK、最小二乘、规划、训练等能力使用 Python 生态，不列入 Mojive 原生功能路线，也不为这些用途预先引入 Eigen。

| C++ 负责的机制 | Python 决定的策略 |
|---|---|
| RHI、渲染、GPU 资源、相机与必要几何计算 | 显示什么、数据如何生成、业务算法如何运行 |
| 稳定的场景 / 帧契约、批量提交、快照与资源寿命 | 场景内容、业务状态和编辑工作流 |
| ImGui 绘制、输入与自定义控件等 UI 基础能力 | 工具行为、面板内容和扩展逻辑 |
| 原生任务执行、同步、取消与完成通知 | 任务选择、业务依赖和处理结果 |
| 原生日志输出、有界历史及订阅机制 | 日志内容、输出配置和业务处理 |

原生层可以按现有 adapter 协议接入 MuJoCo；这不是接管用户的算法或任意可写物理状态。已有 Python `Session` 的状态归属不因基础设施迁移而产生第二个权威副本。

基础设施的接口必须回答以下问题，不能让调用方猜测：

- 数组的 shape、dtype、stride、坐标空间、单位及复制条件是什么？
- 调用返回代表已接收、已提交 GPU，还是结果已可读取？结果如何对应原始帧和相机？
- 数据由谁持有，何时可修改，关闭后哪些对象仍有效？
- 哪些调用需要主线程，哪些释放 GIL，取消和并发调用如何完成？
- 错误通过什么途径报告，队列满时怎样处理，功能不支持时怎样明确反馈？

业务回调不放进原生渲染与物理的关键执行路径；Python 通过批量命令与结果通知扩展。同步 Python UI 回调仍会占用调用线程，需明确执行位置和预算。不能把在任意用户代码下都不卡顿、进程内原生崩溃可完全隔离当作现有保证。

## 已确定的规范

- C++：Qt 风格，`SceneFrame`、`setScene()`、`objectId`、私有成员 `mDevice`，复合文件名 `SceneStream.hpp`。不引入 Qt，不加 `Q` 前缀。
- Python：保留当前 snake_case API，不要求用户把现有脚本改成驼峰。
- 目录：`python/src/mojive` 放 Python 包，`python/tests` 放测试；`cpp` 放自有 C++；`3rdparty` 管理上游源码。
- 默认 C++ 后端：bgfx。标准库契约隔离后端类型，SDL 对比实现保留为可选实验。
- ImGui：已把验证过的 docking 版本原样纳入 Git，保留许可证、上游提交、归档校验与逐文件基线。后续曲率连续圆角的修改可以在此单独提交和审查。
- 其他核心依赖：固定提交的 Git submodule，不追踪移动分支，不在普通构建中自动升级。
- 验证：只做当前本机验证，移除实验 workflow；Linux 后续由你在对应系统上测试。

## 本次准备与下一步的分界

这次完成目录、命名、源码依赖、构建入口和兼容性验证。Python 安装仍使用现有构建方式，**现已通过显式 `renderer="bgfx"` 接入公开 `Renderer`，可用 `make native-viewer` 启动；没有改写已安装的 imgui-bundle wheel**。不会把准备工作的 C++ fixture 当成已经能替代生产 API 的实现。

正式开发已开始：私有 `mojive._native` 扩展已打通场景提交、RGB/深度/分割/ID 读取与释放，并验证数组所有权、异常和 `close()`。公开 Renderer、`out` 缓冲兼容和原有 Viewer 已接入；剩余完整渲染效果与分发仍按[实现与验收清单](cpp-implementation.zh.md)推进。接口以当前 Python 行为为准，不要求用户接触 bgfx handle 或改写成新的 C++ 客户端。

对于用户自己在 Python 中推进的 MuJoCo 仿真，`update_scene(data)` 在调用边界读取稳定状态并形成 Mojive 自有快照；不能把借来的可写 `mjData` / NumPy 指针交给后台线程长期使用。后台渲染与资源任务可以释放 GIL，Python 回调则在持有 GIL 时处理。只有 Mojive 管理仿真时，原生物理 adapter 才独占自己的物理状态；不能擅自接管用户的 `MjData`。

原生代码可用 C++20 的线程、任务队列和显式取消来执行后台工作，但 Python 控制流保持自然：同步 API 继续可用，需要异步时提供清楚的完成结果和生命周期。GUI、CLI 可以共用核心库；当前不要求普通 Python 脚本启动独立 daemon 或经过 IPC。

基础库选型见[依赖与日志方案](cpp-dependencies.zh.md)：首批采用 GLM 和 spdlog，原生 runtime 负责自身日志输出，Python Loguru 可选桥接；Eigen 不进入当前路线，EnTT 仅按基础设施需求评估。

开发命令、命名规则和目录说明见[开发指南](../guides/development.md)，此前性能依据见[后端选型结论](native-backend-decision.zh.md)。
