# 模型配置与切换交互参考

核对日期：2026-11-15。状态：终端内 `/connect` 方案已撤销。本文保留当时的上游源码调研与设计提案，不代表当前功能；当前 `/model` 仅用于选择已有模型，配置方式见 [模型配置](model-configuration.md)。

独立命令入口、清楚的供应商分组、按认证方式显示相关字段，以及可查看当前值的设置总览，都有上游依据。但 OpenCode 仍把连接完成与模型选择串联，Gemini 的设置页则逐项提交修改。Aster 的“基础字段总览、可展开高级项、预览后统一保存、保存后结束”应明确记为本项目的设计选择，不能说成两个产品已经共同采用的流程。[O1][O2][O3][G10]

## 核对范围与版本

只使用官方仓库中的文档和实现，通过 GitHub API 与原始文件核对，没有依据截图或第三方文章推断行为。没有启动这些产品，没有读取真实用户模型配置或凭据，也没有发送模型请求；因此下面的实现行为指静态源码中的控制流。

| 对象 | 固定版本 | 用途 |
| --- | --- | --- |
| OpenCode 已发布版本 | [`v1.19.5`](https://github.com/anomalyco/opencode/releases/tag/v1.19.5)，`810c0eba1a7348036f8bdf61e6632d0c36c6876c`，2026-10-28 | GitHub latest-release 接口在核对时返回此版本；检查 `packages/tui/src/component/dialog-provider.tsx` 和 `dialog-model.tsx`。 |
| OpenCode 开发分支 | [`38bd73789a43bf043599d6196af96583228035d7`](https://github.com/anomalyco/opencode/commit/38bd73789a43bf043599d6196af96583228035d7)，`dev`，2026-11-14 | 检查当前官方文档及新实现；供应商对话框已改为 `dialog-integration.tsx`，不把它与发布版混为一谈。 |
| Gemini CLI | [`4ed06ed0ddc523a6d356a4ad35ddfea331041888`](https://github.com/google-gemini/gemini-cli/commit/4ed06ed0ddc523a6d356a4ad35ddfea331041888)，`main`，2026-11-13 | 检查同一提交下的认证、模型、设置文档和 TUI 命令、组件；不声称这些行为均已发布。 |

## 命令职责

| 产品 | 配置或认证入口 | 模型切换入口 | 两者实际关系 |
| --- | --- | --- | --- |
| OpenCode | TUI `/connect`；CLI `opencode auth login`。无 CLI provider 参数时先打开供应商选择器，可用 `--method key` 指定密钥方式。[O1] | TUI `/models`，注意是复数；当前注册标题为 `Switch model`。[O2] | 文档明确先 `/connect`、再 `/models`。但源码不是严格隔离：独立 `/connect` 成功回调也会打开该供应商的模型选择器；模型选择器另有连接入口。[O2][O5] |
| Gemini CLI | `/auth`，无子命令时默认 `signin`；`login` 是其别名。`signout` 的别名是 `logout`。`/settings` 管理一般设置。[G2][G12] | `/model` 无参数打开模型对话框；`/model manage` 同样打开对话框；`/model set <name> [--persist]` 直接选择；`/model <name>` 也进入 set 分支。[G7] | 认证命令打开 auth 对话框，模型命令打开 model 对话框。设置命令有独立入口；不能因 `/model` 的代码描述写着 “configuration”，就认为它承担任意供应商和 API key 配置。[G2][G7][G12] |

## OpenCode：值得借鉴与需要区分的细节

### 供应商列表有组织，但仍保留完整发现路径

发布版 `providerOptions()` 为少量供应商设置优先级，再按名称和 ID 稳定排序；优先项归入 `Popular`，其余归入 `Providers`，末尾有 `Other / Custom provider`。已连接项显示状态标记。可借鉴的是分组、稳定顺序、易读名称和状态，不是照搬它为自家服务标注的 `Recommended`，或固定相同供应商排序。[O3]

开发版改成 `Popular` / `Services`，列表来自当前 location 的 integration 数据。它展示连接摘要、需要重新登录的状态以及 `Environment only`；已有凭据的供应商打开账号管理，新供应商进入认证方式选择。这是“基于可用方式组织入口”的实例，不是把所有后端配置字段直接列成第一屏。[O4]

### 认证方式决定字段，而不是先询问统一的长表单

发布版只有多个认证方式时才额外询问方式；API key 使用专用输入对话框，OAuth 使用浏览器授权、设备码或授权码流程。`PromptsMethod()` 根据 `when` 条件询问所选方式需要的信息。`Other` 的提示明确说明：这里只保存凭据，使用自定义供应商还需配置 `opencode.json`。因此 OpenCode 的连接对话框不能直接等同于 Aster 的完整模型别名配置器。[O3]

开发版官方文档说明，API key 和 OAuth 方法可以附加 Azure resource、Enterprise domain 等必要字段。实际 `formAnswer()` 逐字段 `await fieldAnswer()`，跳过条件不满足的字段，隐藏字段使用默认值；布尔、选项、文本等类型分别处理。这支持“条件字段和合适控件”的设计，**不支持把一页基础字段与折叠高级字段描述成 OpenCode 已有行为**。[O1][O4]

### 保存和模型切换仍有串联

发布版常规 API key、OAuth 成功后刷新数据，进入按供应商限定的 `DialogModel`。开发版 `connected()` 会刷新 integration、provider、model 数据并提示成功；没有回调时关闭，但应用的 `/connect` 注册实际传入了“打开 `DialogModel`”回调。因此只读 `connected()` 末尾的 `dialog.clear()` 会得出错误结论。[O2][O3][O4]

模型选择器提供当前模型、收藏、最近使用、供应商分组与模糊搜索。开发版选择模型后写入当前选择；若需要选择 variant，随后打开 variant 对话框，否则关闭。它另有 `connect integration` 动作，不能作为“切换界面完全不含配置入口”的现成范例。[O5]

## Gemini CLI：设置总览与模型选择的参考

### 少量认证选项，必要时单独录入凭据

官方认证文档区分 Google 登录、Gemini API key 和 Vertex AI；API key 文档主要介绍环境变量路径。实际 `AuthDialog` 提供这些选项，还根据环境展示 Cloud Shell 或 metadata server 的 ADC 选项，并能受强制认证类型设置约束。它是 Google 产品的认证方式列表，不是多供应商目录。[G1][G3]

选择 Gemini API key 后，当前实现总会进入 key 输入对话框，让用户输入或确认已有 key。`ApiAuthDialog` 有单独的说明、输入框、错误位置和 Enter 提交 / Esc 取消提示；成功认证由 `useAuth` 更新认证状态。可以借鉴“专用凭据交互与局部错误反馈”，不要照搬 Google 专属环境变量、key 字符过滤或认证选项到通用供应商。[G3][G4][G5]

### `/model` 把快捷选择和详细选择分层

文档列出 Auto 与 Manual 路径。源码有 `main` / `manual` 两个视图，Manual 再进入具体模型列表，初始选中项从当前模型推导。可用项受权限和特性开关影响；文档中的具体型号表并不能保证每个运行环境都显示相同列表。[G6][G8]

源码还明确显示 `Remember model for future sessions`，默认 `false`，Tab 切换；选择模型后关闭对话框。直接命令也用 `--persist` 区分是否持久化。这里值得借鉴的是把当前会话选择与未来会话默认值明确区分；不必为 Aster 增加 Google 专属的 Auto 路由或额度展示。[G7][G8]

### `/settings` 有多字段总览，但不是统一确认后保存

`SettingsDialog` 提供搜索、设置标签、说明、当前值、默认值的灰色显示与作用域提示；只提供可写作用域。`BaseSettingsDialog` 负责滚动列表及键盘导航，Enter 对布尔或枚举执行切换，对其他可编辑值进入行内编辑。界面会根据终端高度减少可见项目，说明“同一设置页”不等于强行把所有项目塞进一个屏幕。[G10][G11]

需要区别三个事实：

- 文档把设置分成 General、UI、Advanced 等章节，这不等于 TUI 有一个可展开的 Advanced 面板。[G9]
- 每次切换或编辑提交都会调用 `setSetting()`；没有等待最后一个统一的 Review & Save 按钮。[G10]
- 当前底层组件的编辑模式中，Enter 和 Esc 都会 `commitEdit()`，离开编辑模式后 Esc 才关闭设置页。Aster 若采用草稿与统一保存语义，应自己定义 Esc 的取消/返回行为，不要照搬这个按键副作用。[G11]

## 对 Aster 的六项具体建议

以下是本项目的设计建议，源于用户对独立配置和独立切换的要求；引用说明可借鉴的局部，不表示上游已经实现整套流程。

1. **固定命令职责。** `/connect` 负责供应商、凭据和模型别名配置；`/model` 只切换已配置别名。移除模型列表中的 `Add model…`。空列表显示“尚未配置模型，使用 `/connect` 添加”的说明。保存成功后结束配置并提示可用 `/model` 切换，不自动进入切换器。独立入口参考两者命令设计；最后两点是 Aster 对职责的进一步收紧。[O1][O2][G2][G7]

2. **给供应商目录增加结构，不损失覆盖面。** 首屏突出少量常用供应商，完整目录仍可搜索，并提供适合现有后端的自定义端点路径。用易读名称展示，以 provider ID 作细节；显示已配置或凭据来源等实际已知状态。完整 LiteLLM 目录继续是支持范围依据，前端分组不应变成新的硬编码白名单。分组和状态参考 OpenCode；Aster 的具体分组与常用项需要自行选择。[O3][O4]

3. **把基础配置改为可回看的总览。** 在同一配置页列出别名、供应商、模型 ID、所需凭据和适用的 Base URL；每行显示当前值或默认提示，选中后显示简短说明，Enter 编辑后返回总览。用户可以直接回到某项修正，无需反复走逐字段下一步。多行当前值、说明和行内编辑参考 Gemini 设置页；具体字段来自 Aster 的模型配置职责。[G10][G11]

4. **用高级项收纳少数需要覆盖的参数。** 上下文窗口、回复上限、兼容性参数过滤放入可展开的高级区域；未设置时明确显示自动/默认，按所选供应商和认证方式只展示适用字段。布尔用开关，有限选项用选择器，数字明确单位。类型与条件字段有 OpenCode 依据；“高级区域可展开”和这些字段的分层是 Aster 提案。[O4][G10]

5. **让预览对应一次明确写入。** 编辑阶段保留草稿，先做本地字段验证；预览显示实际将保存的别名、供应商、模型、端点和高级覆盖值，凭据只显示来源或脱敏状态。预览可返回继续编辑，最终 Save 才调用持久化。退出或返回不应意外提交字段。这是 Aster 的统一保存设计；Gemini 的逐项提交和 OpenCode 的逐步认证均不能作为它已存在的证明。[O3][O4][G10][G11]

6. **明确当前选择与默认值的范围。** `/model` 用当前项标记和供应商/模型说明帮助选择，选择完成后关闭。如果需要更改未来会话默认模型，必须有明确的独立选项或动作，不能让“保存一个别名”隐式改变会话模型。作用范围提示参考 Gemini 的 `Remember model for future sessions`；本轮不需要顺带实现它的 Auto 路由、额度 UI 或 OpenCode 的账号管理体系。[G7][G8]

建议流程：

```text
/connect → 选择供应商 → 基础配置总览（可展开高级项）→ 预览 → 保存 → 成功提示并关闭
/model   → 已配置别名列表 → 选择当前会话模型 → 关闭
```

落地时应验证：配置保存不会触发模型切换；切换列表没有写配置入口；基础项可以直接回改且不丢草稿；折叠高级项保留已编辑值；预览不显示原始密钥；取消编辑、返回、取消保存各自有一致且可见的语义。这些是建议验收场景，本调研未运行 Aster 实现测试。

## 一手来源

以下文档链接固定到所核对提交；源码链接可直接查看对应组件、命令及调用链。

[O1]: https://github.com/anomalyco/opencode/blob/38bd73789a43bf043599d6196af96583228035d7/services/site/src/docs/content/cli/providers.mdx
[O2]: https://github.com/anomalyco/opencode/blob/38bd73789a43bf043599d6196af96583228035d7/packages/tui/src/app.tsx#L903
[O3]: https://github.com/anomalyco/opencode/blob/810c0eba1a7348036f8bdf61e6632d0c36c6876c/packages/tui/src/component/dialog-provider.tsx
[O4]: https://github.com/anomalyco/opencode/blob/38bd73789a43bf043599d6196af96583228035d7/packages/tui/src/component/dialog-integration.tsx
[O5]: https://github.com/anomalyco/opencode/blob/38bd73789a43bf043599d6196af96583228035d7/packages/tui/src/component/dialog-model.tsx
[G1]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/docs/get-started/authentication.mdx
[G2]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/commands/authCommand.ts
[G3]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/auth/AuthDialog.tsx
[G4]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/auth/ApiAuthDialog.tsx
[G5]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/auth/useAuth.ts
[G6]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/docs/cli/model.md
[G7]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/commands/modelCommand.ts
[G8]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/components/ModelDialog.tsx
[G9]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/docs/cli/settings.md
[G10]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/components/SettingsDialog.tsx
[G11]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/components/shared/BaseSettingsDialog.tsx
[G12]: https://github.com/google-gemini/gemini-cli/blob/4ed06ed0ddc523a6d356a4ad35ddfea331041888/packages/cli/src/ui/commands/settingsCommand.ts

- **OpenCode 文档**：[连接供应商][O1]。
- **OpenCode 实现**：[命令注册][O2]、[发布版供应商对话框][O3]、[开发版连接对话框][O4]、[开发版模型对话框][O5]。
- **Gemini 文档**：[认证][G1]、[模型选择][G6]、[设置][G9]。
- **Gemini 认证实现**：[命令][G2]、[方式选择][G3]、[key 输入][G4]、[状态更新][G5]。
- **Gemini 模型与设置实现**：[模型命令][G7]、[模型对话框][G8]、[设置命令][G12]、[设置业务逻辑][G10]、[设置布局和键盘交互][G11]。
