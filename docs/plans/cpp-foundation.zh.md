# C++ 正式开发前的准备

**产品仍然是从 Python 使用的 Mojive。** 保留 `mojive.Renderer` 类似 `mujoco.Renderer` 的用法，以及现有 Viewer、SceneRenderer、NumPy 输出和上下文管理方式。C++ 是实现层，负责后续迁入的渲染、批量数据处理、原生任务调度和资源生命周期。

## 已确定的规范

- C++：Qt 风格，`SceneFrame`、`setScene()`、`objectId`、私有成员 `mDevice`，复合文件名 `sceneStream.hpp`。不引入 Qt，不加 `Q` 前缀。
- Python：保留当前 snake_case API，不要求用户把现有脚本改成驼峰。
- 目录：`python/src/mojive` 放 Python 包，`python/tests` 放测试；`cpp` 放自有 C++；`thirdParty` 管理上游源码。
- 默认 C++ 后端：bgfx。标准库契约隔离后端类型，SDL 对比实现保留为可选实验。
- ImGui：已把验证过的 docking 版本原样纳入 Git，保留许可证、上游提交、归档校验与逐文件基线。后续曲率连续圆角的修改可以在此单独提交和审查。
- 其他核心依赖：固定提交的 Git submodule，不追踪移动分支，不在普通构建中自动升级。
- 验证：只做当前本机验证，移除实验 workflow；Linux 后续由你在对应系统上测试。

## 本次准备与下一步的分界

这次完成目录、命名、源码依赖、构建入口和兼容性验证。Python 安装仍使用现有构建方式，**还没有把 bgfx 接进公开 `Renderer`，也没有改写已安装的 imgui-bundle wheel**。不会把准备工作的 C++ fixture 当成已经能替代生产 API 的实现。

下一步先做一个可从 Python 调用的完整纵向功能：用私有 `mojive._native` 扩展承接既有 Renderer 的场景提交、输出与释放流程，检查 RGB/深度/分割、NumPy 输出缓冲、异常和 `close()` 的兼容性，再逐步迁移渲染效果与 UI。接口以当前 Python 行为为准，不要求用户接触 bgfx handle 或改写成新的 C++ 客户端。

对于用户自己在 Python 中推进的 MuJoCo 仿真，`update_scene(data)` 在调用边界读取稳定状态并形成 Mojive 自有快照；不能把借来的可写 `mjData` / NumPy 指针交给后台线程长期使用。后台渲染与资源任务可以释放 GIL，Python 回调则在持有 GIL 时处理。只有 Mojive 管理仿真时，原生物理 adapter 才独占自己的物理状态；不能擅自接管用户的 `MjData`。

原生代码可用 C++20 的线程、任务队列和显式取消来执行后台工作，但 Python 控制流保持自然：同步 API 继续可用，需要异步时提供清楚的完成结果和生命周期。GUI、CLI 可以共用核心库；当前不要求普通 Python 脚本启动独立 daemon 或经过 IPC。

开发命令、命名规则和目录说明见[开发指南](../guides/development.md)，此前性能依据见[后端选型结论](native-backend-decision.zh.md)。
