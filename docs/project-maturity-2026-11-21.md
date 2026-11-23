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
- 实施前：`SKILL.md` 注册为未激活的 `cmd.<id>`；可用技能上下文将 `cmd.*` 视为 slash command 而排除，但编码宿主的 slash 路由只发现 `@slash_command`，`TextSkill` 没有该命令。`/skills` 是只读浏览器。
- 初次验证：离线注册现有文本技能后，已加载 1、已激活 0、宿主 slash commands 0，且技能名不在 `status()` 中。对象仍可能通过 `doc(self)` 被发现，不能据此声称模型完全看不到技能。
- 当前实现：直接输入 `/技能名 [任务]`、在任务中提及 `$技能名`，或由模型按任务从技能简介中选择、按需读取正文。显式调用将完整说明、来源目录和原始任务送入同一回合，保留附件与现有权限；终端提供技能补全。读取技能不自动激活或运行脚本，实施与验收记录见文末。
- 完成标准：复用现有注册和消息路由，支持显式选择与按需读取；验收缺失技能、同名冲突、调用策略、多技能、取消和恢复。浏览说明不自动运行脚本；离线流程验证与真实模型选用效果分别记录。
- 依据：[注册表](../src/aster/skill_registry.py)、[TextSkill](../src/aster/skill.py)、[编码 slash 路由](../packages/aster-cli/src/aster_cli/coding/slash_commands.py)、[终端技能说明](../packages/aster-tui/README.md)。

### U2：接通原生终端的 MCP 配置与连接生命周期

- 类型：确认的能力断点与文档落差；优先级 P1。
- 实施前：TUI 新建和恢复会话均发送空 `mcpServers`；ACP 只消费客户端传入的服务器。框架已有 `.mcp.json` 读取能力，编码宿主未接入；ACP 文档却建议使用 Aster 自己的 `.mcp.json`。
- 已有能力：ACP 已支持 stdio/HTTP/SSE、重复名称处理、单服连接失败警告及部分失败容错。本项复用这些能力，重点补配置加载、优先级和终端状态呈现。
- 完成标准：编码宿主读取受信任的项目/用户配置；明确其与 ACP 客户端转发配置的优先级；连接状态、失败原因与清理可见；用本地 stdio、显式凭据 HTTP、重复名称及断线恢复验收。OAuth 需单独验证。
- 当前实现：ACP 会话读取用户与受信任工作区 `.mcp.json`，按客户端 > 工作区 > 用户整项覆盖；终端 `/mcp` 显示来源、工具数和安全失败原因，并可显式重试。发现和每次调用分别建立并关闭连接，`ready` 表示最近发现或调用成功；本项验收记录见文末。
- 依据：[TUI 控制器](../packages/aster-tui/src/controller.rs)、[ACP MCP 创建路径](../packages/aster-acp/src/aster_acp/server.py)、[框架 MCPManager](../src/aster/mcp/tool.py)、[ACP 说明](../packages/aster-acp/README.md)。

### A1：项目级持续目标执行（候选方向）

- 类型：可选产品方向，尚未确认为需求；不作为当前成熟度阻断项。
- 当前基础：已有单次任务、todo/plan 进度、后台 job、子代理和普通 Python 编排。todo 已包含依赖、完成说明、追加进度日志及快照支持的任务变量；不能把任务持久化列为全新需求。当前 `CodingAgent.handle` 的合同以新收到的用户请求为边界，项目级持续目标的选择、预算与验收合同仍需建立和验证。
- 完成标准：用现有 todo/任务变量保存用户给定的项目目标、受限工作范围、候选任务、优先级和验收条件；每次选择一个有证据的下一任务，执行并验证后更新状态；可暂停和恢复；遇到阻塞时给出具体原因。复用现有存储、进度和工具，不新增独立调度平台。
- 依据：[Todo 及持久任务变量](../src/aster/tools/todo.py)、[CodingAgent.handle](../packages/aster-cli/src/aster_cli/coding/agent.py)、[任务面板范围](../packages/aster-tui/README.md)。
- 如后续确立该产品目标，其前置依赖为 Q1/Q2，以及下面的只读调查、跨回合验证、恢复与预算约束。当前先完成现有编程工作流，不为候选方向新增系统。

### A2：让 Plan 能执行受限的仓库调查

- 类型：当前明确限制；优先级 P1，按编程工作台目标补全。
- 实施前：Plan 拒绝模型生成的 Python/工具执行，因此模型不能自主列目录、搜索或读新文件。用户显式 slash 命令仍经独立宿主路由执行；不能据此把整个会话描述为绝对只读沙箱。
- 当前实现：宿主处理字面量 `self.files.read/list/search` 请求，可调查工作区内的新文件；继续拒绝模型任意 Python、shell、外部工具和写入。真实 ACP 子进程已验证当前仓库调查、关闭与重开；模型响应为离线脚本，真实模型自主选用效果仍需单独验收。见文末 A2 实施记录。
- 完成标准：为模型提供宿主实现的 read/list/search 路径，继续拒绝模型任意 Python、shell 和写入；明确用户显式命令的独立权限合同；验证越界、链接、权限切换与写入拒绝；可完成一次真实仓库调研。
- 依据：[RuntimeControls](../packages/aster-acp/src/aster_acp/runtime_controls.py)、[Plan 范围说明](../packages/aster-tui/README.md)。

### A3：把完成检查扩展为可恢复的任务验收状态

- 类型：现有实现的明确边界；优先级 P1。
- 实施前：`required_checks` 是显式启用、显式文件和当前回合的验证。`Verification.begin()` 清空结果，重新打开后重建初始快照；旧改动没有进一步修改时，需要显式 `verify` 才会触发验证。
- 当前实现：任务身份、文件基线、检查合同和结果随现有快照保存，跨回合和进程重开仍须完成待验收任务；文件、工作区或检查配置改变使旧证据失效。无关问答可保留任务并显示待验收状态，不能借此跳过本回合修改的检查。见文末 A3 实施记录。
- 完成标准：任务级记录待验收的改动和检查合同；跨回合/重启保留未完成状态；代码或检查配置改变后旧结果失效；普通问答不误触发修改任务的全部检查。
- 依据：[verification.py](../packages/aster-cli/src/aster_cli/coding/verification.py)、[现有边界与回归记录](../results/delivery-followup-20261120/README.md)。

### A4：验证副作用后崩溃的任务恢复

- 类型：运行级可靠性缺口；优先级 P1。
- 已有证据：受控 timeout/503 后能重开且不重放旧工具；正常取消能保存当前快照。另一次真实终端强杀保留了磁盘修改与活动记录，却没有当前快照和完成记录。
- 当前实现：宿主先持久化动作意图，再执行生成 Python、shell 或文件写入；执行结果和任务快照保存分别确认。强杀、取消或超时留下不确定动作时，恢复后保留待核对状态，允许受限文件调查，用户通过 `/recovery` 核对后继续；实施与验收见文末。
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

## U1 实施与验收（2026-11-22）

- 实现提交：[6ad1d18a](https://github.com/Ykrsuao/aster-agent/commit/6ad1d18a02c5d6a8d23dc63d404a4dc470835412)，基于 M1a 候选 `25bd8b3d`。质量基线与 M1a 仍由前置草稿 PR #1/#2 交付；本项沿用其提交，不把前置验证计入 U1。
- 使用 `/技能名 [任务]` 或在任务中提及 `$技能名`；也可让模型从名称、完整简介和来源目录中选择相关技能，再调用 `read_text_skill()` 读取正文。显式选择将完整说明、来源目录、字面任务参数与附件送入同一可取消回合；多技能先全部校验，再一次入队，重复别名去重。
- 终端补全只替换光标处的 token，保留前后多行、附件与撤销。每个技能只展示一个主项，支持目录 ID 与完整 registry 名别名；未知命令和首轮命令目录未就绪时保留草稿。内置/Python 命令保留优先级，歧义及禁用的用户调用在模型执行前报错。
- 支持 `user-invocable`、`disable-model-invocation` 和 `agents/openai.yaml` 的 `policy.allow_implicit_invocation`；任一自动调用策略禁用即不进入自动目录，非法布尔配置拒绝发现或保留 reload 前对象。未选择的正文不预加载；读取技能不自动激活或运行脚本，恢复会话重新发现当前文件。
- 用户流程回归覆盖新建/恢复的首轮目录、缺失/冲突/策略、多技能原子入队、字面参数、文本/图片附件、取消清理及恢复后读取新版技能。真实 Rust → ACP 子进程验证直接命令、内联提及、附件与关闭；CodeAct 两步测试验证先读简介再读正文。代码块、混合缩进、转义、路径及 Unicode 名称有独立回归。
- 最终本地 Python 回归：核心技能 **192 passed**；完整 CLI **2,245 passed、41 skipped、1 deselected**；完整 ACP **931 passed、9 deselected、3 xfailed**。新增围栏容器反例修正后，已重新运行受影响的完整 CLI/ACP；定向回归与全套有重叠，不累加。JUnit 保存在本地 `.review-temp/u1-20261122/*-direct-final.xml`。
- 最终 Rust 回归 **643 passed、2 ignored**；fmt、Clippy（`-D warnings`）和 release build 通过。变更范围 11 个 Python 文件的 Ruff 与格式检查通过，全部 **1,348** 个受控 Python 文件通过 SPDX 检查；本地验收目录不属于交付源码。
- 旧交互候选曾在已有 Windows 进程清理测试 `unexpected_backend_exit_stops_unmanaged_descendants` 中出现 `Backend processes started while cleanup was preparing`；后续完整验证通过。本项未修改 `process_tree.rs`，时序根因未证实，不把后续通过记作该竞态已修复。原始与最终日志保存在本地 `.review-temp/`，远端证据以本项最终候选 PR 检查为准。
- 验收边界：模型使用离线替身，未调用真实模型 API；自动选用的实际质量仍需真实任务验证。Plan 模式仍限制模型生成执行，自动读取技能受该限制；显式调用继续走宿主路由。本项没有接入终端长期记忆或更换 embedding 配置。

## U2 实施与验收（2026-11-22）

- 实施分支：`codex/terminal-mcp-lifecycle-20261122`，基于 U1 候选 `d301a262`。质量基线、M1a 和 U1 仍由前置草稿 PR #1/#2/#3 交付；本项的本地与远端验证单独记录。
- 新建与恢复会话读取用户和受信任工作区的 `.mcp.json`，按 ACP 客户端 > 工作区 > 用户整项覆盖。同名无效高优先级配置保留失败，不回退或混合凭据；重复名称、坏文件、禁用配置及单服失败均有可见诊断。stdio 使用会话工作区作为 cwd，参数与环境变量保持字面值，不展开 `${VAR}`。
- 原生终端 `/mcp` 展示来源、transport、状态、工具数和安全失败原因；Ctrl+R 只读刷新，Ctrl+K 或 `/mcp reconnect NAME` 重新读取单服配置并重试。移除、禁用或发现失败会撤下旧工具；前台或后台忙碌时拒绝重试，迟到响应不覆盖新会话或视图。
- 各服务器并发发现，每服限时 30 秒。发现与每次调用分别建立并关闭连接，`ready` 表示最近发现或调用成功，不承诺持续在线。关闭会话会取消并等待发现与调用结束；测试验证阻塞发现、阻塞调用及子进程退出。
- 固定安全错误只暴露可理解原因和 HTTP 状态码，不包含响应正文、URL、header/env 或原始异常链。MCP 对象与回调排除持久快照，恢复按当前配置重建；真实子进程验收证明普通用户状态仍可保存、恢复。
- 本地核心 MCP **313 passed**；registry 两文件 **75 passed**；完整 CLI **2,246 passed、41 skipped、1 deselected**；完整 ACP **964 passed、9 deselected、3 xfailed**。ACP 全套后补入不可信配置元数据读取失败的容错，最终 MCP/workspace trust **59 passed**；registry 的最后 context 清理修复由上述 75 项覆盖。各范围有重叠，不累加。
- 真实 Python ACP → 本地 MCP wire **5 passed、14 deselected**，覆盖 stdio cwd、字面环境值、显式凭据 HTTP、401 脱敏、信任/优先级/重复名称、失败重试、快照恢复和关闭清理。模型使用 FakeLLM；没有连接真实用户服务。
- Windows Rust 全套 **656 passed、2 ignored**；最终后端的 Rust → ACP → 本地 MCP 回归 **1 passed**。Rust fmt、Clippy（`-D warnings`）与 release build 通过；19 个变更 Python 文件（含 CI 后补的记忆测试）的 Ruff/format、差异检查与全部 **1,354** 个受控 Python 文件的 SPDX 检查通过。
- 本地原始证据位于 `.review-temp/u2-20261122/` 与 `.review-temp/u2-rust-20261122/`，不随源码提交。远端 CI、WSL 与 Windows bundle 安装验收须绑定本项最终候选 SHA，并记录在本分支的草稿 PR；前置 U1 成功不代替 U2 结果。Windows seed/reinstall/verify 仍只证明同版本重装。
- 首轮 Windows CI 与安装恢复成功，但包内 `MANIFEST.dirty=true`：既有流程将 memory/security JUnit 写入源码目录的未忽略 `artifacts/`。已把这两份报告移到运行器临时目录，并增加 bundle commit 必须匹配 CI checkout、dirty 必须为 false 的门槛；保留初轮产物，最终以修正后的候选重新验收。
- 第二轮 Windows 在 `test_reconsolidation_allows_reconciler_to_mutate_input_copy` 捕获另一条既有测试的顺序假设：重整按 `(created_at, id)` 排序，测试却把插入首项当作排序首项。同时间戳且 ID 反序的受控探针 5/5 复现，其余三格 15/15 通过。测试现明确输入先后时间，并覆盖两种 ID 顺序；保留更新原 ID、未归档及计数断言，生产排序不变。修后探针 20/20、reflection 回归 32 passed，完整记忆包 **483 passed、13 skipped、1 deselected**。失败与修复验证单独保留，不把第二轮结果当最终候选成功。
- 验收边界：OAuth 不在 ACP 中自动刷新，尚未端到端验收；没有调用真实模型 API、真实用户 MCP 服务或持久凭据。SSE 沿用现有支持，本项新增的完整本地 wire 旅程使用 stdio 与显式凭据 HTTP。
- 用户入口补验复现了忙碌时 `/mcp` 被编辑器提交白名单拦截：此前测试直接调用命令路由，未经过按 Enter 的路径。现放行只读 MCP 列表，新增空闲/忙碌的真实提交入口回归，并让前台/后台忙碌重连拒绝测试也经过该入口。修前用例失败，修后 14 项 MCP 定向测试、fmt 与 Clippy 通过；该修改须以其后续候选 CI 验收，先前成功不覆盖本次修正。

## A2 实施与验收（2026-11-22）

- 实施分支：`codex/plan-readonly-20261122`，基于 U2 候选 `a6deb132`。本项沿用前置草稿 PR #4 的能力与质量基线，独立提交 Plan 调查，不把审计 PR #5 的验证计入本项。
- Plan 每个 cell 接受一个参数为字面量的 `await self.files.read/list/search(...)`。宿主直接派发操作，不解释执行模型 Python、不使用可变的 `agent.files` 或 REPL 绑定；保留原验证、middleware、hooks 和工具输出。源请求或权限版本在准备后变化即拒绝，切回 Manual 后恢复审批。
- 文件边界固定为会话工作区。Linux 保留逐层打开的目录 fd，Windows 保留不允许删除共享的句柄；检查文件身份、实际路径与类型，拒绝越界、符号链接、junction、硬链接文件和特殊文件。Sandbox 路径只映射到本工作区，额外挂载不扩大 Plan 调查范围。
- 读取要求 UTF-8、文件不超过 2 MiB，每次至多 500 行/64K 字符；列表至多 500 项。搜索为区分大小写的字面文本，支持文件 glob，至多 200 个匹配、2000 个条目、16 MiB、64K 输出和 5 秒预算，跳过常见依赖/构建目录。`truncated` 或 `skipped` 代表结果不完整，不能据此断言内容不存在。
- 旧版/新版对照：相同真实 CodeAct 调查用例在 U2 基线拒绝读新文件，在本实现成功返回内容且没有文件修改。真实 ACP 子进程验收使用当前仓库，完成 list/search/read、拒绝写入、关闭与重开；恢复后权限先回到 Manual，再显式切到 Plan 重新调查。
- Windows 完整 CLI **2,277 passed、43 skipped、1 deselected**；最终完整 ACP **990 passed、9 deselected、3 xfailed**。初轮 ACP 发现输入历史测试依赖相邻写入有不同时间戳，已保留真实记录路径、显式设置时间并参数化相反 session ID 顺序，生产排序未改动。另有一次模型会话初始化 30 秒超时，单独复验及上述完整 ACP 复验通过，未放宽超时；根因未证实。
- Linux 定向回归 **182 passed、1 skipped、1 deselected**；补入真实 ACP wire 与输入历史用例后的相关范围 **51 passed、1 skipped**。这些范围有重叠，不相加。Windows 缺少符号链接创建权限及 POSIX FIFO 的用例跳过；独立硬链接与 junction 用例通过，Linux 覆盖符号链接、硬链接和 FIFO，条件跳过 Windows junction。
- Windows Rust 会话控制回归 **61 passed**，fmt check 通过；全仓 Ruff 与格式检查通过，全部 **1,360** 个源码 Python 文件通过 SPDX 检查，差异检查通过。日志与 JUnit 位于本机临时目录 `aster-audit-delivery-da31de6dc7`，不随源码提交；远端结果须绑定本分支最终候选，记录于草稿 PR。
- 最终 A2 候选 `611a232b` 的 [CI](https://github.com/Ykrsuao/aster-agent/actions/runs/45029653122) 与 [WSL full](https://github.com/Ykrsuao/aster-agent/actions/runs/45029656811) 均成功；Linux unit 与 WSL 各 **16,881 passed、56 skipped、584 deselected、3 xfailed**，独立 Linux sandbox、Windows 完整包安装及其余启用检查成功。前一候选曾因 ACP 通知尚未送达而失败；现按本轮独立 ID 等待四个工具终态和最终完整回复，保留原权限、内容及文件未变断言。
- 验收边界：用户显式 slash 命令保留独立宿主权限合同，Plan 不把整个会话或 Python 解释器变成操作系统只读沙箱。模型使用 `FakeLLMClient`，没有调用真实模型 API；真实模型自主调查质量不由这些离线测试证明。A3/A4/A5、M1b 和 R1/R2 仍待推进，A1 保持候选方向。

## A3 实施与验收（2026-11-22）

- 实施分支：`codex/task-verification-20261122`，基于 A2 候选；继续使用显式的 `coding.required_checks`，没有自动猜测测试工具或依赖文件。
- `VerificationState` 保存任务身份、工作区、文件基线、检查合同、通过证据和结果。宿主执行对象、shell 和在途 attempt 不入快照；恢复时绑定当前配置，上一回合仍在运行的检查记为中断，迟到结果不能覆盖新 attempt。
- 任务未验收时，普通 `Done` 仍必须通过全部当前检查；`complete_task: false` 允许没有新修改或新检查请求的问答/规划保留任务。宿主追加待验收提示；流式回复保留同一消息身份，不产生重复回答或错误的中断提示。`NeedInput` 和 `Waiting` 保留任务。
- 每回合开始、执行检查前和接受完成前重载配置。覆盖文件编辑、删除、新建、命令/超时/文件列表变更及工作区迁移均会使旧证据失效；显式删除全部检查禁用门槛并清除通过证据，重新启用必须重新检查。部分有效结果可跨回合和恢复继续使用。
- 增补了显式 Python `return` 的既有验收旁路：计算得到的合法返回值原先只校验类型便结束；现在也执行策略后置条件。拒绝进入可纠正的反馈，检查代码自身出错继续作为基础设施错误报告。单独工具、inline `return_result` 与显式 `return` 都受验收合同约束。
- 反例证据：跨回合未验收改动用例在旧实现失败；流式待验收提示用例复现重复消息/错误中断；显式 `return` 的四条策略用例修前失败，修后策略文件 **17 passed**。流式回复文件 **37 passed**，包含同一预览提交、取消、重试及摘要隔离。
- 持久性验收使用实际 SQLite 保存/重开和两个真实 ACP 子进程：先修改并询问用户，再回答无关问题、关闭进程，随后重开同一会话；提前完成被拒绝，真实宿主检查通过后才能收尾。模型为离线脚本，不代表真实模型自主选用效果。
- Windows Python 3.12 完整 CLI **2,297 passed、43 skipped、1 deselected**，最终完整 ACP **992 passed、9 deselected、3 xfailed**；Linux Python 3.13 全 strategies 加验收、wire、streaming 定向范围 **2,031 passed、124 deselected**，最终完整 ACP **991 passed、1 skipped、9 deselected、3 xfailed**。范围有重叠，不相加；Ruff、格式、全部 **1,362** 个源码 Python 文件的 SPDX 和差异检查通过。
- 初轮 ACP 的三个旧私有状态/恢复断言已更新为公开验收状态合同，并证明重开后的新编辑使已通过结果失效。Windows 另有一次既有模型会话初始化 30 秒超时；三组定向及上述完整复验通过，根因未确认，未扩大上限。测试现在只在失败时附上启动阶段 stderr，保留后续诊断证据。日志与 JUnit 位于本机临时目录 `aster-audit-delivery-da31de6dc7`；远端结果须绑定最终提交记录于草稿 PR。
- 最终 A3 候选 `271aa4c6` 的 [CI](https://github.com/Ykrsuao/aster-agent/actions/runs/45035334353) 与 [WSL full](https://github.com/Ykrsuao/aster-agent/actions/runs/45035341490) 均成功；Linux unit 与 WSL 各 **16,913 passed、56 skipped、584 deselected、3 xfailed**，独立 Linux sandbox、Windows 完整终端包安装及其余启用检查成功。候选验收 harness 现以 `isinstance(result, Done)` 和 `complete_task` 识别已完成的编码任务，覆盖回复但未完成、等待输入、补丁正确但未收尾、收尾但独立验收失败；修前红灯，修后完整文件 **17 passed**。
- 边界：未配置持久存储时只在内存保留；旧版本快照没有历史文件基线可恢复。未列出的文件、任意外部副作用及检查点前强杀不在本项保证内；后者继续归 A4。此门槛防止意外完成声明，不是对任意模型 Python 的安全隔离。A4/A5、M1b 和 R1/R2 仍待推进，A1 保持候选方向。

## A4 实施与验收（2026-11-23）

- 实施分支：`codex/action-recovery-20261122`，基于最终 A3 候选 `271aa4c6`。宿主将动作意图同步写入现有 SQLite 事件后，才执行生成 Python、shell 或文件写入；文件创建在 `O_CREAT` 前记录。journal 保存动作类型、安全摘要、工作目录与结果状态，不新增原始命令、stdin、输出或凭据。
- 执行结果与任务状态保存分开确认。只有当前任务快照保存成功，才能确认它覆盖的已完成动作；保存失败、迟到结果、取消、timeout 与 worker 中断仍需核对。普通可纠正的 `SyntaxError`/`ValueError` 保留执行反馈，不要求每次人工解锁。
- `/recovery` 由持久宿主注册，列出待核对动作；`/recovery ID --note "核对结果"` 仅记录用户检查，不调用模型、不重放动作。用户技能不能抢占这个命令。未核对前拒绝普通 Python、shell、写入及任务完成，仍允许 Plan 同等边界的 `self.files.read/list/search` 调查、提问，以及保留恢复提示的 `complete_task: false` 回复。
- 每回合先绑定本分支的 pre-turn checkpoint 与内容摘要，再使 current snapshot 失效。强杀后缺少 current 时，仅恢复绑定且校验通过的 checkpoint；恢复后继续绑定原持久 storage、事件管理器和宿主执行控制。损坏 journal、来源不符或缺少已确认动作证明的快照拒绝恢复，保留原数据。
- rewind/tree 继续允许导航，但新分支继承当前与目标分支中所选快照未覆盖的动作；晚于快照的完成或人工核对不能替旧状态解锁。冲突的动作/event ID 在写入前拒绝；已有待核对状态不能通过切分支清除。有动作的 provisional 会话保留在历史中，不作为空预备会话清理。
- 真实 ACP 强杀验收覆盖 shell/Python × intent/outcome 四格：非幂等追加并 `fsync` 后强杀进程，重开时拒绝重复追加和提前完成，受限读取仍可用；逐项人工核对后继续，文件始终只有一行。另覆盖真实 native Python 部分写入后 timeout、ACP cancel 后正常 finalize/保存/重开，以及核对后执行、再保存、再次重开。
- Windows Python 3.12 完整 CLI **2,389 passed、43 skipped、1 deselected**，最终完整 ACP **1,005 passed、9 deselected、3 xfailed**；Linux Python 3.13 完整 CLI **2,426 passed、6 skipped、1 deselected**，最终完整 ACP **1,004 passed、1 skipped、9 deselected、3 xfailed**。四格强杀包含在完整 ACP 内，范围不重复相加。全仓 Ruff、格式、全部 **1,372** 个源码 Python 文件 SPDX 与差异检查通过。
- 初轮完整 ACP 的九个命令清单断言补入了宿主 `/recovery`，保留技能 metadata、顺序和路由断言。Windows 另有一次模型会话初始化 30 秒超时，启动日志停在 `provider_code_cache`；本地离线缓存探针冷建 **12.891 秒**、复用 **0.047 秒**，上述完整 ACP 复验通过，未扩大超时，未确认该次耗时变化的根因。较早 Python intent 强杀用例的一次会话读取失败未再复现；测试保留底层读取诊断，最终两平台四格均通过。日志与 JUnit 位于本机临时目录 `aster-audit-delivery-da31de6dc7`，远端结果绑定最终候选后记入草稿 PR。
- 最终 A4 候选 `1417a44c` 的 [CI](https://github.com/Ykrsuao/aster-agent/actions/runs/45042013985) 与 [WSL full](https://github.com/Ykrsuao/aster-agent/actions/runs/45042018982) 均成功。Linux unit 与 WSL 各 **17,018 passed、56 skipped、584 deselected、3 xfailed**；Windows 完整终端包安装/重装、Linux 原生终端、Linux sandbox 及其余启用检查通过，macOS 条件任务未启用。
- 边界：一个 Python cell 与内部 shell 可能分别产生记录，用户须核对全部列出的 ID。记录只能表明宿主观察到的执行边界，不能推断任意代码的全部外部效果、自动回滚、阻止绕过框架的任意程序，或保证副作用恰好执行一次。模型均为离线脚本，未调用真实模型 API；A5、M1b 与 R1/R2 的集成和验收继续独立推进，A1 保持候选方向。

## A5 实施与验收（2026-11-23）

- 实施分支：`codex/task-budget-20261122`，基于最终 A4 候选 `1417a44c`。显式 `coding.task_budget` 提供调用数、token 与累计活动时间限额，默认关闭；复用实际 Completion/Responses 客户端和 admission 边界。契约与示例见 [任务预算](task-budget.md)。
- 宿主在 admission 成功后、provider 派发前持久化调用预留，每次框架重试单独计数。排队拒绝和排队取消不花调用；预算绑定时拒绝 SDK 隐式重试。token 按供应商实际报告的输入/输出结算，不重复加入缓存和 reasoning 子集；缺失、错误或中断 usage 保持未知，有 token 限额时阻止继续派发。未知费用保留未知，明确的零费用仍为已知零。
- SQLite 的独立 `task_budgets` ledger 不随聊天快照、回合、模型切换、摘要或恢复退款。后台工作、子代理、摘要和 Python 技能共享 owner；活动时间按并发区间的并集累计。强杀后保守计入未结束区间直至恢复，时钟回退导致时间未知时停止任务。
- `/task-budget` 可查看原因，`/task-budget new` 显式创建新额度；有活动任务或未结算请求时拒绝重置。rewind/fork 复制最新消费并停为 `forked_task`，不能利用旧快照退款。预算耗尽后仍可用宿主 `/recovery` 核对；豁免按实际绑定的宿主方法识别，同名用户技能继续受限。
- 联动反例已修复：预算耗尽曾拦住恢复管理（三种限额均复现）；部分 Python 写入后嵌套模型调用到限，正常保存曾清掉不确定动作。现在预算中断保留待核对 journal，保存或新额度都不能清掉它。使用真实 SessionStore、实际文件追加和离线 CompletionClient 验证，修前红灯、修后通过。
- Linux 完整 CLI **2,440 passed、6 skipped、1 deselected**，完整 ACP **1,005 passed、1 skipped、9 deselected、3 xfailed**；UnifiedLLM/storage/任务预算相关核心范围 **3,923 passed、53 deselected**。Windows 完整 CLI **2,403 passed、43 skipped、1 deselected**，最终完整 ACP **1,006 passed、9 deselected、3 xfailed**；最终预算/恢复/配置并发/模型选择与缓存关键范围 **77 passed**。各范围有重叠，不相加。
- 原 Linux deadline 反例和新增的 `run`/`run_stream` 取消用例证明子进程在取消后继续写入。控制等待现在复用已有进程树中断恢复，并在两路输出消费的 `finally` 回收 reader。Linux Bash/Shell 与预算范围 **497 passed**；当前集成后的 Bash/配置并发回归 **24 passed**，验证无迟到追加、只启动一次、原有后台任务存活和同会话立即复用。
- Windows 首轮的两处失败分别是 SDK 缓存冷建占用协议等待，以及配置测试在第二个进程仍准备 SDK 时判断锁竞争。模型选择现在先在隔离目录准备真实安装 SDK 的可复用缓存，保持每次协议请求的 **30 秒**上限；专用冷启动测试仍从空缓存验证 loader、资源和原生扩展。配置测试把就绪信号移到实际取锁前，使用显式 Responses 配置避免无关 SDK 冷导入，保留所有原超时和数据断言。离线缓存 profiling 为冷建 **14.245 秒**、复用 **0.043 秒**，不把测试范围调整称为生产冷启动提速。
- 全仓 Ruff、格式及差异检查通过，SPDX 检查通过 **1,382** 个源码 Python 文件。日志与 JUnit 保存在本机临时目录 `aster-audit-delivery-da31de6dc7`；草稿 PR 的远端检查绑定其最终候选。
- 最终 A5 候选 `439d0488` 的 [CI](https://github.com/Ykrsuao/aster-agent/actions/runs/45108439348) 与 [WSL full](https://github.com/Ykrsuao/aster-agent/actions/runs/45108443201) 均成功。Linux unit 与 WSL 各 **17,120 passed、56 skipped、584 deselected、3 xfailed**；Windows 完整包安装、Linux 原生终端、Linux sandbox 及其余启用检查通过。
- 边界：响应可超过剩余 token 额度，后续调用才停止；本项不是供应商硬 token 上限或最终账单。async deadline 不能抢占阻塞事件循环的同步 Python，有时间限额时同步 provider 派发会预先拒绝。预算对象跨独立进程、guest 自建模型客户端或未结算的 broker 导出显式拒绝；任意用户程序直连供应商不在框架计量范围。模型均为离线替身，M1b 与 R1/R2 继续独立验收，A1 保持候选方向。

## M1b 实施与验收（2026-11-23）

- 实施分支：`codex/project-memory-20261122`，基于 A5 候选 `439d0488`。默认关闭；只有受信任项目的 `.aster/settings.yaml` 显式配置，才挂载现有 `MemorySkill`/`MemoryManager`。每个 canonical workspace 使用独立 SQLite 库，数据库名与路径受宿主约束，检查数据库、目录与 WAL 的符号链接。使用方式见 [CLI 项目记忆](../packages/aster-cli/README.md)。
- 提供 remember/recall/search 与 `/memory status/list/view/delete/off/reload/rebuild`。宿主负责 manager 生命周期，快照不保存开放连接或执行对象；新建/恢复会话重新绑定当前项目。不开启原始事件自动保存或后台反思，关闭后卸载 hooks、提示上下文和 SQLite 句柄。
- `/memory off` 原子写回项目设置，现有 manager 和已取得的 store/事务句柄也不能继续读写。删除后不再检索；其他项目的记录保持隔离。更换 embedding 模型或维度要求显式重建到新库，保留原库且不自动切换；模型别名和 `api_key_env` 复用已有配置，未新增凭据。
- 与预算联动的反例已修复：到限后五种本地管理动作曾全部被拦住；远程 embedding 检查只依赖 ContextVar，直接宿主操作曾绕过已有 session owner。现在 status/list/view/delete/off 仅按绑定的宿主方法和 action 放行，不调用模型、不退款；reload/rebuild 与同名用户技能继续受限。远程 embedder 同时检查当前 scope 和 agent 的持久 owner，已绑定任务预算时明确要求 hashing embedder。
- Windows 接入实现阶段完整 CLI **2,433 passed、47 skipped、1 deselected**；接到最终 A5 后，预算/恢复/记忆管理联合范围 **71 passed、4 skipped**，最终完整 ACP **1,007 passed、9 deselected、3 xfailed**。Linux Python 3.13 最终完整 CLI **2,474 passed、6 skipped、1 deselected**；完整 ACP 与记忆包联合范围 **1,487 passed、16 skipped、10 deselected、3 xfailed**，其中 ACP **1,006 passed**，记忆包 **481 passed**。范围重叠，不相加。
- 真实 ACP 离线旅程覆盖 trusted project 启用、写入、关闭进程、重开检索、不同项目隔离、删除、持久关闭、恢复后的 hook/连接清理与重新启用。Windows 本机不能创建符号链接的四项用例明确跳过，Linux 完整 CLI 已实际通过这四项；没有把 Windows 跳过当作通过。
- 全仓 Ruff、格式与差异检查通过，SPDX 检查通过 **1,386** 个源码 Python 文件。初轮联合测试中旧 mock 整块替换预算模块导致四项失败，现只替换所需符号；保持真实 owner/ledger 路径，单文件 **30 passed、4 skipped**，后续联合与完整协议验证通过。日志与 JUnit 位于本机临时目录 `aster-audit-delivery-da31de6dc7`。
- 最终 M1b 候选 `3f0325e6` 的 [CI](https://github.com/Ykrsuao/aster-agent/actions/runs/45110932561) 与 [WSL full](https://github.com/Ykrsuao/aster-agent/actions/runs/45110936623) 均成功。Linux unit 与 WSL 各 **17,155 passed、56 skipped、584 deselected、3 xfailed**；Windows 完整包安装、Linux 原生终端、Linux sandbox 及其余启用检查通过。
- 平台边界限定为已验证的本地 hashing/NumPy/SQLite 项目记忆和 native Windows/Linux 宿主流程；未调用真实 embedding 或模型 API，未验收真实 Chroma 服务或全部可选后端，不据此扩大通用 memory 包的平台声明。记忆收益与过期事实判断归 R2，发行安装/跨版升级仍归 R1；A1 保持候选方向。

## R1 实施与验收（2026-11-23）

- 实施分支：`codex/release-evidence-20261122`，基于最终 M1b 候选 `3f0325e6`。同一 final tag、版本、干净 SHA 与一次候选构建绑定 wheel/sdist、完整 Windows ZIP、安装/重装、跨版恢复和任务报告的字节哈希。发布端只消费已经验收的产物，不再重新构建；门槛和渠道说明见 [RELEASING.md](../RELEASING.md)。
- 普通 CI 不要求历史旧包或真实模型报告；只有显式 `release_candidate=true` 才进入严格候选验收。候选 SHA 必须可达默认分支，旧包必须来自本仓库已发布的 final release，具有 API digest 与完整 manifest，且版本严格增加。缺少材料时拒绝严格候选，不伪造同版本重装为跨版升级。
- checker 从 GitHub API 核验仓库与 head-repository、workflow 路径/ID、dispatch run、当前 attempt、同一 SHA、必需成功 jobs 和 artifact digest。十任务报告要求独立完成/代码验收、固定预算、实际模型五键 identity/fingerprint、fixture/runner 哈希及可信来源 run/attempt；拒绝 fork、旧 attempt、过期或篡改产物。下载重定向不转发 token，解包校验路径、链接和体积。
- 发布继续沿用 `release: published` 到 PyPI、手工 dispatch 到 TestPyPI；未新增 tag push 自动发布或凭据。发布说明精确指定一个 `Candidate artifact: ID`。runner 与冻结 fixture 的 Git 属性固定为 LF，保证 Windows checkout 的源码字节哈希与冻结 Git blob 一致。
- Windows Python 3.12 离线 gate、terminal release smoke、make-release 合同及候选 harness 共 **237 passed**；Linux Python 3.13 同范围 **233 passed、4 skipped**，四项跳过属于 Windows 专用安装旅程。全仓 Ruff、格式、**1,388** 个源码 Python 文件 SPDX 与差异检查通过。未执行 make_release 主流程、publish workflow、真实模型、tag 或 release 操作。
- 最终 R1 候选 `dfe0e1f6` 的 [CI](https://github.com/Ykrsuao/aster-agent/actions/runs/45118812407) 与 [WSL full](https://github.com/Ykrsuao/aster-agent/actions/runs/45118816888) 均成功。Linux unit 与 WSL 各 **17,275 passed、56 skipped、584 deselected、3 xfailed**；Windows 完整终端包安装/重装、Linux 原生终端、Linux sandbox 及其余启用检查通过。严格发行候选与 macOS 条件任务未启用。
- 完成边界：门槛实现与离线验收已完成，端到端发行验收仍未完成。可信 `.github/workflows/terminal-acceptance.yml` 真实任务 producer 尚未提供，历史已发布完整 Windows 包及真实跨版本安装仍缺材料；当前严格 gate 因而保持拒绝。离线替身、旧 dirty 包或同版本重装不能代替这些材料。

## R2 计量、来源与离线对照验收（2026-11-23）

- 实施分支：`codex/eval-evidence-20261122`，基于最终 R1 候选 `dfe0e1f6`。从 raw spans、TraceExplorer、worker 结果到实验 JSON/JSONL，分别保存已知 token 小计与完整性；缺失用量和费用保持未知，明确的零仍为已知零。各指标保留观察数与未知数，不把未知算成成功或零费用。
- 磁盘 worker 的本地 JSONL exporter 现在独立于 OTLP 初始化；实际子进程的零用量、部分用量与未知用量贯穿 raw spans、worker JSON 和实验 JSONL。受限 worker 在结果输出前卡于无需执行的 `tracemalloc.clear_traces()` 已独立复现；该进程随即退出，移除清理后原失败与非有限 timeout 探针 **4 passed**，保留原 **30 秒**协议上限和实际内存监控。
- 终端验收记录实际五键模型 identity/fingerprint、候选 SHA/dirty/diff、fixture/runner 字节哈希、预算与 repeat，分别判断代理完成和独立代码验收。严格聚合拒绝身份、来源或限制不可比较及记录不完整的结果；NaN/无穷 timeout 在模型调用前拒绝。源码与 fixture 固定 LF，保持 Windows checkout 字节与 Git blob 一致。
- 新增 [离线记忆对照](../experiments/memory_evidence/README.md)：实际 MemoryManager、MemorySkill、SQLite、NumPy 与 hashing embedder，六个合成任务各重复两次，分别运行 off/current/stale/corrected 四组。FakeLLM 脚本从真实检索复制事实并引用实际 ID，不含 fixture 答案；独立 scorer 判定答案、完整任务、引用与过期事实采用。
- 原始报告来自 native Windows/Python 3.12 的 `5f728da2`，测量时间 **2026-11-23 13:47 +08:00**，共 **48 次任务观察和 48 次 FakeLLM 接口调用**，无调用失败或未知评分；current 与 corrected 各 **12/12** 完整任务，off 为 **0/12**，stale 为 **0/12** 且 **12/12** 采用过期事实。更正组执行 **24 次**显式更新；token/费用为 `null`，完整性为 false，外部供应商调用为零。
- 报告保留真实 dirty 标记、空 tracked diff 与空 untracked-source 集合；四个源码与 fixture 的 SHA 已验证对应同一 Git 字节，runner/策略/模型指纹均验证通过。仅保存判定与哈希；六个原始问题和 24 个新旧事实值未进入报告。旧实验结果保留在提交历史，不将离线对照当作干净发行验收。
- Windows eval_pipeline 全目录、TraceExplorer token 与终端验收 **364 passed、47 skipped**；记忆 harness 与 README **10 passed**。范围有重叠，不相加。默认 testpaths 现已纳入 eval_pipeline，Windows 默认收集 **17,736/18,320** 项、**584 deselected**，包括新增 worker/usage 回归；真实模型测试仍默认跳过，须显式开启 `RUN_MODEL_TESTS`。
- Linux 首轮关联范围 **873 passed、51 skipped、1 failed**，唯一失败是超时 span 测试启用了缺席 OTLP collector。独立复现确认任务按 1 秒限额取消，随后网络导出与同步等待超过 30 秒协议上限；用例现使用本地 JSONL，保留原 **30 秒**和全部 span 断言，生产 worker 未因此改动。修后 Windows/Linux timeout 与 worker 各 **22 passed**；完整 Linux Python 3.13 关联复验 **874 passed、51 skipped**。这些范围有重叠，不相加；首轮失败日志保留。
- 独立审查未发现指定计量/来源链路的新增阻断问题；全仓 Ruff、格式、全部 **1,395** 个源码 Python 文件 SPDX 与差异检查通过。日志、JUnit 与报告完整性核验保存在本机 `aster-audit-delivery-da31de6dc7` 临时目录，远端结果在草稿 PR 中绑定最终候选。
- 完成边界：本项固化计量、可比较来源与离线记忆存取/接线/评分对照；未运行真实付费模型，也没有新的真实模型重复可靠率、自动反思或学习收益。真实仓库/终端旅程的可信 producer、同一候选的真实模型重复基线和 R1 跨版本发行材料仍待补齐；A1 保持候选方向。
