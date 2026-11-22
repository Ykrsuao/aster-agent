# Aster 项目完整性评估与推进清单

评估日期：2026-11-21。源码基线：`a7b546ebf9146abfcb88cd1e1b05fc4281b8a8bf`。

复核日期：2026-11-22。已重新核对当前源码、远端 CI 和离线复现；主要方向成立。此次细化已有能力的边界，补入 WSL 配置缺口，并将已复现的记忆库完整性问题从终端记忆新功能中拆出，提升优先级。实施进展见文末。

## 评估目标与结论

以当前 README 的定位为准：Aster 同时提供终端 AI 编程工作台和 Python Agent 框架；支持 Linux 与原生 Windows。预期使用方式是个人开发者在真实仓库中持续使用，也允许同一进程运行多个代理。此次没有把云端 SaaS、团队账号、技能市场或 macOS 支持设为必需目标。

项目已经具备较多底层能力。当前主要欠缺三类工作：把框架能力接成可控的用户流程；让长任务在失败、重启和验证后形成完整交付；用同一候选版本的持续测试和发布验收证明这些流程可靠。

“项目还缺什么”不能直接等同于 TODO 数量或没有某个命令。下面区分当前缺陷、已确认的能力断点、证据不足，以及需要设计的产品能力。没有为项目编造完成百分比。

本会话由开发代理主动盘点、排序和推进，是协作方式；这不自动构成给 Aster 增加“自主经营整个项目”功能的产品需求。下文 A1 因此单列为候选方向，不进入当前必做顺序。

## 已有基础

| 范围 | 已有能力 | 评估时不能误判为缺失的内容 |
| --- | --- | --- |
| Agent 框架 | 类型合同、Predict/CodeAct、普通 Python 编排、子代理、上下文与事件 | 不需要另造图式工作流框架 |
| 持久化 | SQLite 会话、快照、检查点、分支、显式恢复示例 | 已有持久化基础，但不等于任意副作用后都能自动续跑 |
| 终端 | 会话、草稿、附件、暂停队列、steering、任务和后台 job、工具输出、Git diff、文件回退 | 基础交互并非空白；仍需验证完整用户旅程 |
| 权限与执行 | 工作区信任、权限模式、宿主工具审批、可选 microVM | Python sandbox 与宿主工具权限是不同边界 |
| 扩展 | 文本/Python 技能、激活和热重载、MCP stdio/HTTP/SSE | 底层支持不代表原生终端已有正常配置和调用入口 |
| 记忆 | remember/recall/search/forget、反思、SQLite 与多种向量后端 | 终端目前没有挂载这套长期记忆；HUD memory 是主机 RAM |
| 质量工具 | Python/Rust 测试、CI、安全检查、安装包构建、eval_pipeline、真实任务试运行 | 测试数量不能代替同版本端到端成功率 |

## 优先推进项

### Q1：恢复当前提交的 Windows 交付验证

- 类型：当前质量问题；优先级 P0。
- 现状：评估时当前 HEAD 的 CI run `44908884044` 失败于 `windows-core-smoke` 的 Rust terminal regressions。失败日志显示首条 `cargo fmt --check` 即报告格式差异并退出；不能表述为 Rust 逻辑测试失败，后续 clippy、Rust 测试、构建及 Windows 完整包安装验收未执行。该 run 的 Linux unit、sandbox、Python 3.13、build、frontend、安全扫描和 Rust audit 并非全部失败。
- 影响：本地通过的回归不能证明当前候选提交在干净 Windows CI 环境中可交付。
- 完成标准：定位失败测试或检查；保留失败证据，修复真实原因；对同一候选提交完成 Rust 检查、完整包构建及仓库外安装/升级验收。
- 依据：[Windows CI](../.github/workflows/ci.yml)、[发布验收要求](../RELEASING.md)、[当前提交的 CI run](https://github.com/Ykrsuao/aster-agent/actions/runs/44908884044)。远端状态是 2026-11-21 的观察，不是永久结论。

### Q2：补齐已承诺组件与平台的持续回归覆盖

- 类型：确认的测试范围缺口；优先级 P0。
- 初评现状：根 `testpaths` 包含 CLI/ACP/bench，但没有 `packages/aster-memory/tests`；常规根目录 pytest 因此不能作为记忆包回归的证明。持续 CI 的 Rust terminal fmt/clippy/test/build 位于 Windows job；Linux Python 与 sandbox 成功不能证明 Linux 原生终端成功。
- 复核补充：WSL full 脚本排除 sandbox extra 后仍执行 sandbox 测试，quick 分支还引用已删除的 `tests/test_windows_platform.py`。旧基线 `ca768beb` 的 WSL run `44896837576` 中，宿主组 15,842 passed，sandbox 组 283 failed；代表性原因是缺少 `cloudpickle`。该配置到评估基线未变，但这些数量不能解释为当前 HEAD 的 283 个逻辑缺陷。
- 完成标准：记忆包测试进入一个明确、安装了对应依赖的 CI job；Linux 原生终端执行 Rust 检查与必要的启动/关闭回归；WSL 的宿主覆盖合同、安装依赖和实际执行命令一致，真实 microVM 由独立 Linux sandbox job 验证；记录实际收集的测试范围。
- 依据：[pytest 配置](../pyproject.toml)、[CI](../.github/workflows/ci.yml)、[WSL CI](../.github/workflows/wsl.yml)。

### U1：让文本技能从“可浏览”变成“可明确使用”

- 类型：确认的能力断点；优先级 P1。
- 现状：`SKILL.md` 注册为未激活的 `cmd.<id>`；可用技能上下文将 `cmd.*` 视为 slash command 而排除，但编码宿主的 slash 路由只发现 `@slash_command`，`TextSkill` 没有该命令。`/skills` 当前是只读浏览器。
- 验证：离线注册现有文本技能后，已加载 1、已激活 0、宿主 slash commands 0，且技能名不在 `status()` 中。对象仍可能通过 `doc(self)` 被发现，不能据此声称模型完全看不到技能。
- 完成标准：提供一个明确的文本技能调用流程，复用现有注册和消息路由；用户能选择技能并将其用于任务；验收缺失技能、同名冲突、取消和重连。浏览说明不自动运行脚本。
- 依据：[注册表](../src/aster/skill_registry.py)、[TextSkill](../src/aster/skill.py)、[编码 slash 路由](../packages/aster-cli/src/aster_cli/coding/slash_commands.py)、[终端技能说明](../packages/aster-tui/README.md)。

### U2：接通原生终端的 MCP 配置与连接生命周期

- 类型：确认的能力断点与文档落差；优先级 P1。
- 现状：TUI 新建和恢复会话均发送空 `mcpServers`；ACP 只消费客户端传入的服务器。框架已有 `.mcp.json` 读取能力，编码宿主未接入；ACP 文档却建议使用 Aster 自己的 `.mcp.json`。
- 已有能力：ACP 已支持 stdio/HTTP/SSE、重复名称处理、单服连接失败警告及部分失败容错。本项复用这些能力，重点补配置加载、优先级和终端状态呈现。
- 完成标准：编码宿主读取受信任的项目/用户配置；明确其与 ACP 客户端转发配置的优先级；连接状态、失败原因与清理可见；用本地 stdio、显式凭据 HTTP、重复名称及断线恢复验收。OAuth 需单独验证。
- 依据：[TUI 控制器](../packages/aster-tui/src/controller.rs)、[ACP MCP 创建路径](../packages/aster-acp/src/aster_acp/server.py)、[框架 MCPManager](../src/aster/mcp/tool.py)、[ACP 说明](../packages/aster-acp/README.md)。

### A1：项目级持续目标执行（候选方向）

- 类型：可选产品方向，尚未确认为需求；不作为当前成熟度阻断项。
- 当前基础：已有单次任务、todo/plan 进度、后台 job、子代理和普通 Python 编排。todo 已包含依赖、完成说明、追加进度日志及快照支持的任务变量；不能把任务持久化列为全新需求。当前 `CodingAgent.handle` 的合同以新收到的用户请求为边界，项目级持续目标的选择、预算与验收合同仍需建立和验证。
- 完成标准：用现有 todo/任务变量保存用户给定的项目目标、受限工作范围、候选任务、优先级和验收条件；每次选择一个有证据的下一任务，执行并验证后更新状态；可暂停和恢复；遇到阻塞时给出具体原因。复用现有存储、进度和工具，不新增独立调度平台。
- 依据：[Todo 及持久任务变量](../src/aster/tools/todo.py)、[CodingAgent.handle](../packages/aster-cli/src/aster_cli/coding/agent.py)、[任务面板范围](../packages/aster-tui/README.md)。
- 如后续确立该产品目标，其前置依赖为 Q1/Q2，以及下面的只读调查、跨回合验证、恢复与预算约束。当前先完成现有编程工作流，不为候选方向新增系统。

### A2：让 Plan 能执行受限的仓库调查

- 类型：当前明确限制；优先级 P1，按编程工作台目标补全。
- 现状：Plan 拒绝模型生成的 Python/工具执行，因此模型不能自主列目录、搜索或读新文件。用户显式 slash 命令仍经独立宿主路由执行；不能据此把整个会话描述为绝对只读沙箱。
- 完成标准：为模型提供宿主实现的 read/list/search 路径，继续拒绝模型任意 Python、shell 和写入；明确用户显式命令的独立权限合同；验证越界、链接、权限切换与写入拒绝；可完成一次真实仓库调研。
- 依据：[RuntimeControls](../packages/aster-acp/src/aster_acp/runtime_controls.py)、[Plan 范围说明](../packages/aster-tui/README.md)。

### A3：把完成检查扩展为可恢复的任务验收状态

- 类型：现有实现的明确边界；优先级 P1。
- 现状：`required_checks` 是显式启用、显式文件和当前回合的验证。`Verification.begin()` 清空结果，重新打开后重建初始快照；旧改动没有进一步修改时，需要显式 `verify` 才会触发验证。
- 影响：一次任务跨多个回合或恢复后，产品没有统一的“还有哪些改动尚未验收”的状态。已有当轮验证和并发隔离不能当作这个流程已经完成。
- 完成标准：任务级记录待验收的改动和检查合同；跨回合/重启保留未完成状态；代码或检查配置改变后旧结果失效；普通问答不误触发修改任务的全部检查。
- 依据：[verification.py](../packages/aster-cli/src/aster_cli/coding/verification.py)、[现有边界与回归记录](../results/delivery-followup-20261120/README.md)。

### A4：验证副作用后崩溃的任务恢复

- 类型：运行级可靠性缺口；优先级 P1。
- 已有证据：受控 timeout/503 后能重开且不重放旧工具；正常取消能保存当前快照。另一次真实终端强杀保留了磁盘修改与活动记录，却没有当前快照和完成记录。
- 完成标准：先对宿主可观测的“外部副作用已发生、内部进度未保存”动作保存可辨认状态；恢复时先核对结果未知的动作，避免无条件重复；故障注入覆盖非幂等追加之后、检查点之前强杀。恢复聊天历史与恢复正在执行的任务分别验收，不承诺任意 Python/shell 副作用都能自动恰好执行一次。
- 依据：[显式检查点语义](concepts/prompts-and-context.md)、[持久工作流示例](../examples/advanced/durable_hitl.py)、[强杀记录](../results/daily-use-20261120/README.md)、[正常取消和重开](../results/daily-finish-20261120/README.md)。

### A5：统一自主任务累计预算

- 类型：需要建立和验证的契约；优先级 P1，设计项。
- 已有：iteration/tool/cell 限制、重试、broker 生命周期内跨进程共享的 provider-attempt 上限，以及 bench 使用的任务 token 累计器。它们不能直接证明已有跨子代理、摘要、重试及恢复持久共享的任务 token/费用预算；后续应复用已有计量组件。
- 完成标准：先提供任务调用数/token/时间预算，所有参与任务的请求计入同一范围；到限后保存并说明原因；恢复不重置已消耗额度；未知价格显示未知，估算金额不等同供应商最终账单。
- 依据：[CodeAct 配置](../src/aster/config/strategy_config.py)、[admission 范围](concepts/llm-admission-control.md)。本轮没有复现预算失控。

### M1a：保护记忆库的 embedding 身份与维度

- 类型：已复现的库级检索完整性缺陷；优先级 P1，Q2 后可独立修复，不依赖终端记忆功能。
- 初评现状：向量存储没有绑定 embedding backend/model/dim 身份；同维换模型无法识别不同向量空间。库里已有数据库 schema 版本迁移，缺的是 embedding 兼容合同。
- 复现：默认 NumPy 后端的临时 SQLite 库存入 2 维向量后，以 3 维重开查询产生矩阵错误；继续写入 3 维记录仍会成功持久化，之后以 2 维重开查询也因混合维度失败。两条文本仍在，未证实文本或关系丢失；其他向量后端有各自约束，不能套用此复现结果。
- 完成标准：读取和写入前校验 embedding 身份及维度，拒绝不兼容写入且保留原数据；提供保留文本/关系的显式重建路径；验收维度改变、同维换模型、既有数据库、重开及迁移中断。
- 依据：[MemoryStore](../packages/aster-memory/src/aster_memory/store.py)、[向量后端](../packages/aster-memory/src/aster_memory/vector_backends.py)、[MemoryManager](../packages/aster-memory/src/aster_memory/manager.py)。
- 实施状态：库级修复与本地验收已完成，见文末 M1a 实施记录；终端记忆仍归 M1b。

### M1b：接入可选的终端项目记忆

- 类型：确认尚未集成的产品能力；优先级 P2，依赖 Q2/M1a。
- 现状：`CodingAgent` 有意跳过 `aster.memory`，把作用域和存储责任留给宿主；ACP 尚未安装长期记忆。已有同会话上下文恢复不能代替跨会话项目记忆。
- 完成标准：增加可选的项目记忆挂载与查看/删除/关闭流程；验收不同项目隔离、跨会话检索、删除后不再检索、换模型与迁移中断。
- 依据：[CodingAgent](../packages/aster-cli/src/aster_cli/coding/agent.py)、[宿主责任测试](../packages/aster-cli/tests/test_coding_agent.py)、[可复用示例](../examples/quickstart/12_memory.py)。
- 平台边界：memory README 仍声称原生 Windows 不支持；入口导入或离线单元测试成功不足以证明完整支持。接入主产品前需按完整记忆流程验收，再修正文档。

### R1：让发布物、候选验收和升级渠道对应同一版本

- 类型：确认的交付流程缺口；优先级 P1，可与功能工作并行。
- 现状：已有 Windows bundle、哈希、仓库外安装和 dev16→dev18 升级证据；根 README 明确没有公开下载渠道。普通 Python wheel 不包含原生终端；现有 bundle 构建针对 Windows。`publish.yml` 已检查五包版本与发布 tag 一致，并在干净环境检查导入、版本和帮助命令；缺的是完整候选验收与发布门槛绑定。Windows CI 的 seed/reinstall/verify 只证明同版本重装，不能代替跨版本升级。
- 完成标准：选择维护者授权的分发位置；同一干净候选 SHA 的构建产物通过测试、完整终端安装、旧版本升级和新任务验收，才进入发布；公开标明支持平台与源码安装边界。不能用历史 dirty 候选包的证据代替新提交验收。
- 依据：[根 README](../README.md)、[发布流程与已知 gate 边界](../RELEASING.md)、[publish.yml](../.github/workflows/publish.yml)、[dev18 交付证据](../results/delivery-release-20261120/README.md)。

### R2：扩展并固化已有自主交付基线，补齐记忆效果验证

- 类型：证据缺口；优先级 P1，持续贯穿后续工作。
- 现状：已有 eval_pipeline、冻结十任务、独立验证器、预算、结果记录和退出判定。十任务曾在 8 次调用预算下 6/10 通过，在 12 次调用预算的完整候选运行中 10/10 通过；判定同时要求代理 Done 与代码验收通过。这是单模型单次结果，不是长期可靠率，也不能合并不同预算的结果。三项真实仓库试运行虽通过功能后验，但一项未 Done、一项遗漏格式；后续真实 TUI 检查仍未在预算内自主收尾。供应商失败与审批等待均有记录，不能全归因产品或模型。
- 记忆边界：文档引用的效果评测 harness 明确未发布在本仓库，无法据此持续复现文档中的收益。
- 完成标准：扩展现有固定任务与独立验收到真实仓库和终端旅程；同一候选版本、模型与预算下重复运行，分别记录补丁正确、检查通过、自主收尾、人工干预、恢复、耗时/token 和未知费用；保留失败样本。记忆增加 on/off 和过期事实对照。
- 依据：[eval_pipeline](../util/eval_pipeline/README.md)、[真实任务记录](../results/real-work-trial-20261120/README.md)、[终端收尾记录](../results/daily-finish-20261120/README.md)、[记忆效果声明](../packages/aster-memory/src/aster_memory/README.md)。

## 建议实施顺序

1. Q1/Q2：先建立可信的当前基线，修复实际失败并补齐持续回归范围。
2. M1a：先修复已复现的记忆库兼容问题；不等待终端记忆新功能。
3. U1/U2/A2：把技能、外部工具和只读调查接成用户可用流程；各项可独立验证和交付。
4. A3/A4/A5：补齐任务验收、恢复和累计预算合同，先证明一项长任务能验证、保存并恢复，再扩大并发或无人值守时长。A1 保留为候选方向。
5. M1b：完成平台验收，再接入有明确项目边界和用户控制的可选长期记忆。
6. R1/R2：持续积累同候选验收；满足安装、任务、恢复、升级标准后交付稳定候选。

每项完成后，应把真实提交和验收结果补到该项；新发现进入同一清单。新增功能不会自动提高成熟度，只有该功能的完整用户流程与失败路径完成验收，才能更新状态。

## 初评验证与边界（2026-11-21）

- 检查了当前源码、关键调用关系、文档、CI 配置、远端当前提交的 CI 状态及已有真实任务/安装/恢复记录。
- 子代理执行了文本技能离线注册自检、记忆入口导入和内存向量维度自检；未调用真实模型或付费服务。
- 本轮没有重跑全仓测试、真实终端/microVM 或新的供应商任务；没有将历史测试数量当作当前 HEAD 已通过的证明。
- 本轮产物是评估与推进清单，尚未实施以上功能。不能保证此清单穷尽所有缺陷；设计项需要以最小原型与独立验收进一步收敛。

## 复核与推进（2026-11-22）

- 三个独立审查分别核对了用户流程、任务可靠性/记忆、发布/评测。结论支持继续推进，以上已合入范围和优先级修正。
- 初次离线验证：记忆包 353 passed、13 optional-backend skipped、1 integration deselected；skills/slash/MCP/Plan 定向回归 45 passed、1 既有 xfailed；任务验收与宿主记忆责任回归 21 passed；冻结十任务的损坏种子全部拒绝、参考解全部通过。
- 额外探针复现了跨回合未验收改动放行（A3）和默认 NumPy 后端混维写入后检索失败（M1a）。原始文本保留；未调用真实模型或付费服务。
- 本批已实现 Q1 的 Rust 格式修复，以及 Q2 的默认记忆测试收集、Windows 记忆回归、Linux 原生终端 job 和 WSL 宿主配置修复。首轮 Windows fmt/Clippy、628 项 Rust 测试与 160 项交付/启动/ACP 回归已通过；首轮默认测试收集选中 16,359 项，其中记忆包 366 项。
- 新增的 Windows 记忆 CI 在候选 `f9e2be2a` 捕获一项测试时间假设：连续创建的新旧事实可能同时间，随机 ID 决定排序。冻结时钟可复现；测试现已显式设置先后时间，并覆盖两种 ID 顺序，保留原断言。修复后本地记忆回归 354 passed、13 skipped、1 integration deselected；默认收集选中 16,360 项，其中记忆包 367 项。生产排序未改动，M1a 仍未修复。失败证据见 [CI 44967334424](https://github.com/Ykrsuao/aster-agent/actions/runs/44967334424/job/137039226272)。
- 最终完成状态还应关联同一干净候选的安装/升级与 CI 结果，后续证据汇总在 [质量基线 PR](https://github.com/Ykrsuao/aster-agent/pull/1)。远端旧 SHA 的失败与新候选验证须分别记录；同版本重装不得记作跨版本升级。

## M1a 实施与验收（2026-11-22）

- 实现提交：[3c0673b5](https://github.com/Ykrsuao/aster-agent/commit/3c0673b55a59c7a0b944d25f66e20ab6cb9bffbd)，基于质量基线 `aa08ee4c`。以上初评和基线验证描述保留为历史记录，不代表此提交仍允许混维写入。
- 数据库现在保存 embedding 身份、维度和索引 namespace。具名 embedder、raw 向量和无身份旧库不能凭相同维度互相认领；查询与 embedding 写入拒绝身份/维度不符、错误形状和非有限数值。已有库未声明身份时仍可查看文本、关系及原始向量，不代表数据库绝对只读。
- 新增显式 `rebuild_embeddings()`：从一致快照保留文本、完整元数据、ID、关系、维护历史及归档记录，重新生成全部向量后发布到新文件。目标已存在、并发创建、失败或中断均不覆盖原库或目标；重建不自动切换宿主配置。说明及示例见 [记忆库兼容与重建](../packages/aster-memory/src/aster_memory/README.md#embedding-compatibility-and-rebuilding)。
- 旧版/新版对照探针：2 维库重开后尝试写入 3 维记录，`aa08ee4c` 接受并留下 2 条记录；新实现抛出 `EmbeddingMismatchError`，只留下原有 1 条记录，原文本均保留。
- 本地最终回归：记忆包与 `test_coding_agent.py` 共 **498 passed、13 skipped、2 deselected**；临时安装真实 sqlite-vec 的定向回归 **94 passed、9 skipped**。两组有重叠，不相加。后者的跳过均依赖未安装的 Chroma；其客户端交互、同服务隔离及复制库回归使用离线替身验证。
- 独立审查补入并验证 stale 句柄、事务回滚、迁移中断清理、索引初始化恢复，以及同一 Chroma 服务下不同路径的数据库副本隔离。跨主机同 namespace、同绝对路径的独立副本仍需使用不同 `VectorConfig.collection`。
- 变更范围 Ruff、格式及差异检查通过；全部 1,347 个受控 Python 文件（含新增回归）通过 SPDX 检查。原始全工作区 SPDX 命令会遇到既有验收目录内刻意生成的非 UTF-8 文件；该目录保留，不纳入交付。JUnit 保存在本地 `.review-temp/m1a-20261122/`，不随源码提交，远端证据以此分支的 PR 检查为准。
- 升级边界：打开库会升级到 schema v3，旧版库代码拒绝打开，降级前需保留原始备份；重建发布要求支持硬链接的文件系统。未调用真实 embedding API，未验收真实 Chroma 服务，未接入终端长期记忆，也未扩大记忆包的完整平台支持声明。
