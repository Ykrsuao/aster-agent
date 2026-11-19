// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Capture actual Ratatui cells for visual review, without starting a backend.
//! Conversation content and usage notifications are synthetic offline fixtures.
use aster_tui::controller::{Action, Controller};
use aster_tui::state::{Entry, Overlay, Phase, Session, State, Tool};
use aster_tui::ui;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::{backend::TestBackend, style::Color, Terminal};
use serde_json::{json, Value};
use std::time::{Duration, Instant};
use unicode_width::UnicodeWidthStr;

fn color(value: Color, fallback: &str) -> String {
    match value {
        Color::Rgb(r, g, b) => format!("#{r:02x}{g:02x}{b:02x}"),
        _ => fallback.into(),
    }
}

fn preview_image() -> aster_tui::attachments::Attachment {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, 96, 48);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        let mut pixels = Vec::new();
        for y in 0..48 {
            for x in 0..96 {
                let color = if (12..84).contains(&x) && (10..38).contains(&y) {
                    if y < 16 {
                        [147, 186, 133]
                    } else if x < 40 {
                        [64, 64, 64]
                    } else {
                        [200, 200, 200]
                    }
                } else {
                    [20 + x as u8, 20 + y as u8, 32]
                };
                pixels.extend_from_slice(&color);
            }
        }
        writer.write_image_data(&pixels).unwrap();
    }
    aster_tui::attachments::Attachment::from_image("terminal-layout.png", &bytes).unwrap()
}

fn preview_context_sections(category: &str) -> Vec<Value> {
    let sections = match category {
        "system_prompt" => vec![(
            "system_prompt",
            "<system_prompt>\nYou are Aster, a coding agent working in the current workspace.\n\nRead repository instructions before changing files.\nPreserve the user's drafts and unrelated local changes.\nUse the available tools to inspect files and run checks.\nReport what changed and how it was verified.\n\nExecution workflow:\n1. Inspect the relevant source and current state.\n2. Apply the requested change.\n3. Check keyboard, mouse and narrow-window behavior.\n4. Run the affected tests.\n5. Explain how to open the result.\n\nKeep Chinese text and original newlines intact.\n</system_prompt>",
        )],
        "system_tools" => vec![(
            "API tool schemas",
            "{\n  \"name\": \"execute_python\",\n  \"parameters\": {\n    \"code\": {\"type\": \"string\"}\n  }\n}",
        )],
        "memory_files" => vec![(
            "repository_instructions",
            "<repository_instructions>\nRun focused tests after changing behavior.\nPreserve the user's uncommitted work.\n</repository_instructions>",
        )],
        "skills" => vec![("skills", "<skills>\nAvailable workflow instructions.\n</skills>")],
        "messages" => vec![(
            "Conversation text",
            "User: 我想查看系统提示词和模型输出。\n\nAssistant tool call:\nexecute_python(code=\"print(6 * 7)\")\n\nTool result:\n42\n\nUser: 保留中文、换行和长文本滚动。",
        )],
        "runtime_context" => vec![(
            "execution_context",
            "<execution_context>\nExecution backend: native\nWorking directory: workspace/aster\n</execution_context>",
        )],
        "other" => vec![("Additional context", "<context>\nRequest framing.\n</context>")],
        "output" => vec![
            ("Assistant text", "You can now open the system prompt, conversation text and model output from /context."),
            ("Tool calls", "[\n  {\n    \"name\": \"return_result\",\n    \"arguments\": {\n      \"result\": {\n        \"message\": \"查看正文已接上。\",\n        \"explanation\": \"System prompt, messages and output are selectable.\"\n      }\n    }\n  }\n]"),
        ],
        _ => vec![],
    };
    sections
        .into_iter()
        .map(|(key, text)| {
            json!({
                "key": key, "text": text, "characters": text.chars().count(), "truncated": false
            })
        })
        .collect()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let width: u16 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(120);
    let height: u16 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(36);
    let requested_mode = args.get(3).map(String::as_str).unwrap_or("work");
    let selecting_text = requested_mode.ends_with("-selected");
    let mode = requested_mode
        .strip_suffix("-selected")
        .unwrap_or(requested_mode);
    let mut state = State::new(
        "workspace/aster".into(),
        "global:deepseek-v4.1-flash".into(),
        false,
    );
    state.phase = Phase::Ready;
    state.monochrome = false;
    state.status = "Ready".into();
    if !matches!(
        mode,
        "welcome"
            | "welcome-hud-full"
            | "sent-hud-full"
            | "starting"
            | "starting-multiple"
            | "starting-paused"
            | "welcome-drafts"
            | "drafts"
            | "drafts-empty"
    ) {
        state.title = "Polish the terminal workspace".into();
        state.entries=vec![
            Entry::User("把终端界面升级成完整的工作台，保留中文输入和会话恢复。".into()),
            Entry::Assistant("I'll inspect the existing layout, then update the conversation view and run the interaction tests.".into()),
            Entry::Tool(Tool{id:"read".into(), title:"Read src/ui.rs".into(),status:"completed".into(),
                content:"Loaded 248 lines from src/ui.rs".into(),diff:String::new(),expanded:false,
                started:Some(Instant::now()),elapsed:Some(0.4)}),
            Entry::Tool(Tool{id:"edit".into(),title:"Update conversation layout".into(),status:"completed".into(),
                content:"Applied patch to src/ui.rs".into(),diff:"--- a/src/ui.rs\n+++ b/src/ui.rs\n@@ -18,2 +18,3 @@\n-    render_prompt(frame);\n+    render_conversation(frame, &state);\n+    render_composer(frame, &state);\n     render_status(frame);".into(),
                expanded:true,started:Some(Instant::now()),elapsed:Some(1.2)}),
            Entry::Tool(Tool{id:"test".into(),title:"Run interaction tests".into(),status:"in_progress".into(),
                content:"running 20 tests\ntest controller::chinese_input ... ok\ntest controller::resume_session ... ok\ntest transport::cancel_request ... ok".into(),diff:String::new(),
                expanded:false,started:Some(Instant::now()-Duration::from_secs(4)),elapsed:None}),
        ];
        state.phase = Phase::Busy;
        state.status = "Running interaction tests…".into();
        state.started = Some(Instant::now() - Duration::from_secs(12));
        state.editor.insert_str("完成后检查窄窗口下的布局");
        state.sessions = vec![
            Session {
                id: "session-current".into(),
                title: "Polish the terminal workspace".into(),
                updated: "2026-11-09".into(),
            },
            Session {
                id: "session-previous".into(),
                title: "Fix ACP protocol output".into(),
                updated: "2026-11-08".into(),
            },
        ];
        state.session_id = Some("session-current".into());
    }
    match mode {
        "starting" | "starting-multiple" | "starting-paused" => {
            state.phase = Phase::Preparing;
            state.queued.push_back("hi".into());
            if mode == "starting-multiple" {
                state.queued.push_back("Check the current project".into());
                state.queued.push_back("Summarize the changes".into());
            }
            state.queue_paused = mode == "starting-paused";
            state.hud.configure("full").unwrap();
        }
        "permissions" | "permissions-plan" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.runtime_supported = true;
            state.mode = if mode == "permissions-plan" {
                "plan"
            } else {
                "code"
            }
            .into();
            state.approval_policy = "ask".into();
            state.runtime = json!({
                "permissionMode":if mode == "permissions-plan" { "plan" } else { "manual" },
                "mode":state.mode,
                "approval":"ask"
            });
            state.open(Overlay::Permissions);
        }
        "reasoning" | "reasoning-unknown" | "reasoning-unsupported" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.runtime_supported = true;
            state.runtime = json!({
                "reasoningLevel":"high", "reasoningSelection":"high",
                "reasoningDefault":"medium", "reasoningLevels":["low", "medium", "high"]
            });
            if mode == "reasoning" {
                state.model = "declared-demo".into();
            }
            if mode != "reasoning" {
                state.runtime["reasoningLevel"] = Value::Null;
                state.runtime["reasoningSelection"] = Value::Null;
                state.runtime["reasoningDefault"] = Value::Null;
                state.runtime["reasoningLevels"] = if mode == "reasoning-unknown" {
                    Value::Null
                } else {
                    json!([])
                };
            }
            state.open(Overlay::Reasoning);
        }
        "model-effort" | "model-effort-unknown" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.runtime_supported = true;
            state.model = "declared-demo".into();
            state.runtime = json!({
                "reasoningLevel":"medium", "reasoningSelection":null,
                "reasoningDefault":"medium", "reasoningLevels":["low","medium","high"],
                "models":[
                    {"id":state.model, "name":"Declared demo"},
                    {"id":"custom-demo", "name":"Custom demo", "reasoningLevels":["quick","deep"], "reasoningDefault":"quick"},
                    {"id":"unsupported-demo", "name":"Unsupported demo", "reasoningLevels":[]}
                ]
            });
            if mode == "model-effort-unknown" {
                state.model = "deepseek-flash".into();
                state.runtime["models"][0] = json!({"id":state.model});
                state.runtime["reasoningLevel"] = Value::Null;
                state.runtime["reasoningSelection"] = Value::Null;
                state.runtime["reasoningDefault"] = Value::Null;
                state.runtime["reasoningLevels"] = Value::Null;
            }
            state.open(Overlay::Models);
            if mode == "model-effort" {
                state
                    .model_effort_drafts
                    .insert(state.model.clone(), Some("high".into()));
            }
        }
        "model-switch" | "model-switch-empty" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.model_selection.supported = true;
            state.runtime = json!({"models":[]});
            if mode == "model-switch" {
                state.runtime["models"] = json!([
                    {"id":state.model}, {"id":"official"}, {"id":"work-claude"}
                ]);
            }
            state.open(Overlay::Models);
        }
        "context" | "context-all" | "context-pending" | "context-system" | "context-messages"
        | "context-output" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.open(Overlay::Context);
            state.context_expanded = mode == "context-all";
            state.terminal_graphics = aster_tui::terminal_input::GraphicsCapabilities {
                sixel: true,
                cell_pixels: aster_tui::terminal_image::CellPixels::new(10, 23),
                sixel_display_mode: Some(false),
            };
        }
        "tree" | "tree-search" | "tree-empty" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.editor = Default::default();
            let nodes = if mode == "tree-empty" {
                vec![]
            } else {
                vec![
                    json!({"id":"u1","parentId":null,"role":"user","preview":"Review the parser","navigable":true}),
                    json!({"id":"a1","parentId":"u1","role":"assistant","preview":"Two approaches are available","navigable":true}),
                    json!({"id":"u2","parentId":"a1","role":"user","preview":"Try a small patch","navigable":true}),
                    json!({"id":"a2","parentId":"u2","role":"assistant","preview":"The patch passes its tests","navigable":true}),
                    json!({"id":"u3","parentId":"a1","role":"user","preview":"尝试另一条分支","navigable":true}),
                    json!({"id":"a3","parentId":"u3","role":"assistant","preview":"Alternative plan ready","navigable":true}),
                ]
            };
            let mut response = json!({
                "nodes":nodes,
                "currentNodeId":if mode == "tree-empty" { Value::Null } else { json!("a3") }
            });
            if mode == "tree-search" {
                response["query"] = json!("small patch");
                response["matchedNodeIds"] = json!(["u2"]);
            }
            state.session_tree =
                aster_tui::session_tree::SessionTree::from_response(&response).unwrap();
            state.tree_search_supported = true;
            state.open(Overlay::Tree);
            if mode == "tree-search" {
                state.query = "small patch".into();
                state.selected = 2;
            } else {
                state.selected = state.choices().len().saturating_sub(1);
            }
        }
        "welcome-drafts" | "drafts" | "drafts-empty" => {
            use aster_tui::drafts::{provisional_scope, DraftSnapshot, LoadedDraft};

            if mode != "drafts-empty" {
                state.saved_drafts = vec![
                    LoadedDraft {
                        session_id: Some(provisional_scope("preview")),
                        revision: 2,
                        snapshot: DraftSnapshot {
                            composer: "Review 中文 parser".into(),
                            queued: vec!["Check the tests".into()],
                            queue_paused: true,
                            ..DraftSnapshot::default()
                        },
                        warning: None,
                    },
                    LoadedDraft {
                        session_id: Some("saved-preview-session".into()),
                        revision: 1,
                        snapshot: DraftSnapshot {
                            composer: "Update documentation".into(),
                            in_flight: Some("Inspect the startup path".into()),
                            queue_paused: true,
                            ..DraftSnapshot::default()
                        },
                        warning: None,
                    },
                ];
            }
            if mode != "welcome-drafts" {
                state.open(Overlay::Drafts);
            }
        }
        "steering" => {
            state.steering_supported = true;
            state.entries.push(Entry::Notice { text: "Correction received · waiting for the next model request\n先检查取消和重连，不要改动其他模块。".into(), error:false });
            state.queued.push_back("下一轮再补文档和截图".into());
        }
        "files" => {
            state.file_candidates = vec![
                "packages/aster-tui/src/controller.rs".into(),
                "packages/aster-tui/tests/session_controls.rs".into(),
                "docs/contributing.md".into(),
            ];
            state.open(Overlay::Files);
        }
        "attachments" | "image" => {
            state.attachments.push(aster_tui::attachments::Attachment::Text {
                name:"src/controller.rs".into(), uri:"file:///workspace/src/controller.rs".into(),
                text:"pub fn reconnect(&mut self) {\n    self.queue.pause();\n    self.restore_saved_session();\n}\n".into(),
            });
            state.attachments.push(preview_image());
            state.open(Overlay::Attachments);
            if mode == "image" {
                state.selected = 1;
            }
        }
        "approval" | "approval-expanded" | "approval-fallback" => {
            use aster_tui::state::{ApprovalOption, ApprovalPreview, ApprovalRequest};
            state.mode = "code".into();
            state.approval_policy = "ask".into();
            state.status = "Waiting for execution approval".into();
            state.approvals.push_back(ApprovalRequest {
                id:json!(7), title:"Run shell command?".into(),
                code:"result = await self.shell.run(\n    'cargo test --locked --test session_controls'\n)\nprint(result)\n".into(),
                preview: (mode != "approval-fallback").then(|| ApprovalPreview {
                    summary: "Run shell command".into(),
                    details: vec!["Command: cargo test --locked --test session_controls".into()],
                    complete: true,
                }),
                show_code: mode != "approval",
                options:vec![
                    ApprovalOption { id:"allow_once".into(), name:"Allow once".into(), kind:"allow_once".into() },
                    ApprovalOption { id:"allow_session".into(), name:"Allow all code for this session".into(), kind:"allow_always".into() },
                    ApprovalOption { id:"reject".into(), name:"Reject execution".into(), kind:"reject_once".into() },
                ],
            });
            state.open(Overlay::Approval);
            state.selected = state.approvals[0].default_selection();
        }
        "tasks" | "progress" => {
            state.tasks_supported = true;
            state.tasks = json!({
                "steps":[
                    {"id":"step-1","title":"Wire steering into the active turn","status":"completed"},
                    {"id":"step-2","title":"Verify draft recovery and approvals","description":"Run deterministic tests against the local backend.","status":"in_progress","blockedBy":[],"comments":["The composer remains editable while tests run."]},
                    {"id":"step-3","title":"Build the native release executable","status":"pending","blockedBy":["step-2"]}
                ],
                "currentTask":"Verify draft recovery and approvals",
                "waitReason":"Waiting for the test process",
                "jobs":[{"id":"job-tests-3","channel":"tests","title":"Interaction tests","status":"running","canStop":true,"output":"running 20 tests\nattachments ... ok\napproval_cancel ... ok\nreconnect ... ok"}]
            });
            state.status = "Verify recovery · waiting for test process".into();
            if mode == "tasks" {
                state.open(Overlay::Tasks);
                state.selected = 3;
            }
        }
        "recovery" => {
            state.phase = Phase::Failed;
            state.started = None;
            state.status = "Backend disconnected · /reconnect · /retry".into();
            state.recovery = Some("继续检查取消和重连的边界条件".into());
            state.queued.push_back("补充文档和使用示例".into());
            state.queue_paused = true;
            state.entries.push(Entry::Notice {
                text: "Backend disconnected. Input was saved; /retry restores it for review."
                    .into(),
                error: true,
            });
        }
        "rollback" => {
            use aster_tui::workspace_changes::{ChangeStage, ChangedFile};
            state.phase = Phase::Ready;
            state.started = None;
            let file = ChangedFile { path:"src/controller.rs".into(), original_path:None,
                status:"Modified".into(), stage:ChangeStage::Unstaged,
                patch:"diff --git a/src/controller.rs b/src/controller.rs\n--- a/src/controller.rs\n+++ b/src/controller.rs\n@@ -1 +1 @@\n-let queue_paused = false;\n+let queue_paused = true;\n".into() };
            state.rollback.hunks = aster_tui::rollback::hunk_choices(&file).unwrap();
            state.rollback.file = Some(file);
            state.open(Overlay::Rollback);
            state.selected = 1;
        }
        "python-error" => {
            use aster_tui::controller::Controller;

            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.title = "Inspect a failed Python call".into();
            state.editor = tui_textarea::TextArea::default();
            state.entries.clear();
            let mut controller = Controller::new(state, None, false);
            controller.message(json!({
                "method":"session/update","params":{"sessionId":"session-current","update":{
                    "sessionUpdate":"tool_call_update","toolCallId":"python-error",
                    "title":"Python failed","status":"failed","_meta":{"aster":{"elapsedSeconds":0.01}},
                    "content":[
                        {"type":"content","content":{"type":"text","text":"```python\nimport subprocess\nresult = subprocess.run(['git', 'status', '--short'])\n```","_meta":{"aster":{"format":"fenced-code"}}}},
                        {"type":"content","content":{"type":"text","text":"```text\nCell In[1], line 1\n    import subprocess\n    ^\nimport of 'subprocess' is blocked. This module can freeze the event loop and is not allowed in agent code.\n```","_meta":{"aster":{"format":"fenced-code"}}}}
                    ]
                }}
            }));
            controller.toggle_tool("python-error");
            state = controller.state;
        }
        "tools" => state.open(Overlay::Tools),
        "commands" => state.open(Overlay::Commands),
        "slash" => {
            state.editor = tui_textarea::TextArea::default();
            state.editor.insert_str("/");
        }
        "changes" => {
            use aster_tui::workspace_changes::{ChangeStage, ChangedFile, WorkspaceChanges};
            state.workspace_changes = Some(WorkspaceChanges {
                root: state.workspace.clone().into(),
                files: vec![
                    ChangedFile { path:"src/ui.rs".into(), original_path:None, status:"Modified".into(), stage:ChangeStage::Staged,
                        patch:"--- a/src/ui.rs\n+++ b/src/ui.rs\n@@ -18,2 +18,3 @@\n-    render_prompt(frame);\n+    render_conversation(frame, &state);\n+    render_composer(frame, &state);\n     render_status(frame);".into() },
                    ChangedFile { path:"src/controller.rs".into(), original_path:None, status:"Modified".into(), stage:ChangeStage::Unstaged,patch:"--- a/src/controller.rs\n+++ b/src/controller.rs\n@@ -1 +1 @@\n-old\n+new".into() },
                    ChangedFile { path:"docs/使用说明.md".into(), original_path:None, status:"Added".into(), stage:ChangeStage::Untracked,patch:"@@ -0,0 +1 @@\n+使用右键粘贴，按 Enter 发送。".into() },
                ], notice:None,
            });
            state.open(Overlay::Changes);
        }
        "sessions" => state.open(Overlay::Sessions),
        "sidebar" => state.show_sidebar = true,
        "search" => {
            state.open(Overlay::Search);
            state.query = "layout".into();
        }
        "queue" => {
            state.queued.push_back("检查中文输入和长回答滚动".into());
            state
                .queued
                .push_back("完成后运行回归测试，并总结这轮改动".into());
            state.open(Overlay::Queue);
        }
        "markdown" => {
            state.phase = Phase::Ready;
            state.status = "Ready".into();
            state.started = None;
            state.entries = vec![Entry::Assistant(
                "## Terminal workspace\n\nRead **clearer answers**, keep `code` readable, and find earlier output.\n\n> Your draft stays intact while you browse.\n\n- [x] Search messages and tool output\n- [x] Send follow-up messages in order\n- [ ] Review the final result\n\n```rust\nfn main() {\n    println!(\"你好，Aster\");\n}\n```\n\n| Feature | Shortcut |\n| --- | --- |\n| 搜索对话 | Ctrl+F |\n| 消息队列 | Ctrl+E |\n| 工具输出 | Ctrl+T |".into())];
        }
        "compose" => {
            state.phase = Phase::Ready;
            state.started = None;
            state.editor = tui_textarea::TextArea::default();
            state.editor.insert_str("请检查 src/main.rs 中的会话恢复逻辑，保留现有草稿和中文输入，完成后运行回归测试并说明改动。长输入现在会自动折行，方向键按屏幕上的行移动。");
        }
        "compact" => {
            state.phase = Phase::Ready;
            state.started = None;
            state.editor = tui_textarea::TextArea::default();
            state.entries = vec![
                Entry::User("请检查终端的输入、会话和工具执行流程。\n要求：\n- 保留中文和多行草稿\n- 长日志支持紧凑预览\n- 搜索能定位隐藏的内容\n- 运行测试并汇报结果".into()),
                Entry::Assistant("I'll check the input flow and run the regression tests.".into()),
                Entry::Tool(Tool {
                    id: "compact-test".into(), title: "Run cargo test".into(), status: "completed".into(),
                    content: format!("running 24 tests\n{}\n\ntest result: ok. 24 passed; 0 failed", (1..=24).map(|i| format!("test interaction_{i:02} ... ok")).collect::<Vec<_>>().join("\n")),
                    diff: String::new(), expanded: true, started: Some(Instant::now()), elapsed: Some(1.2),
                }),
                Entry::Assistant("The input flow is ready. Text stays unchanged while the composer wraps it on screen.\n\n```rust\nfn visible_rows(text: &str, width: usize) -> usize {\n    text.chars().count().div_ceil(width.max(1))\n}\n```".into()),
            ];
        }
        _ => {}
    }
    if mode.starts_with("selection-") {
        state.phase = Phase::Ready;
        state.status = "Ready".into();
        state.started = None;
        state.show_sidebar = false;
        state.editor = tui_textarea::TextArea::default();
        state.entries = vec![
            Entry::User("聊天正文也支持拖选，松开后就复制。".into()),
            Entry::Assistant("Reply: Drag to copy **visible text**, including 中文 and 👩‍💻.".into()),
        ];
        match mode {
            "selection-tool" | "selection-tools" => {
                state.entries = vec![Entry::Tool(Tool {
                    id: "selection-tool".into(),
                    title: "Read source output".into(),
                    status: "completed".into(),
                    content: "Copied tool output:\nalpha\t中文 👩‍💻 e\u{301}\nsecond line kept"
                        .into(),
                    diff: "--- a/source.rs\n+++ b/source.rs\n@@ -1 +1 @@\n-before\n+after".into(),
                    expanded: true,
                    started: None,
                    elapsed: Some(0.4),
                })];
                if mode == "selection-tools" {
                    state.open(Overlay::Tools);
                }
            }
            "selection-compose" => {
                state
                    .editor
                    .insert_str("输入框：拖选就复制 中文 👩‍💻\n保留这行草稿");
            }
            "selection-help" => state.open(Overlay::Help),
            "selection-diagnostics" => {
                state
                    .diagnostics
                    .push_back("Diagnostic output: 中文 👩‍💻\nsecond line".into());
                state.open(Overlay::Diagnostics);
            }
            _ => {}
        }
    }
    let has_sample_usage = state.session_id.is_some() && mode != "context-pending";
    if has_sample_usage {
        let mut update = json!({
            "sessionUpdate": "usage_update", "used": 18000, "size": 100000,
            "cost": {"amount": 0.0024, "currency": "USD"}
        });
        match mode {
            "context" | "context-all" | "context-system" | "context-messages"
            | "context-output" => {
                update["used"] = json!(32000);
                update["size"] = json!(200000);
                update["_meta"] = json!({"aster": {"context": {
                    "version": 1, "model": "openai/deepseek-v4.1-flash",
                    "requestId": "preview-request", "outputAvailable": true,
                    "windowTokens": 200000, "reservedOutputTokens": 32000,
                    "attribution": "characters",
                    "categories": [
                        {"id": "system_prompt", "tokens": 2000},
                        {"id": "system_tools", "tokens": 9500},
                        {"id": "memory_files", "tokens": 3000},
                        {"id": "skills", "tokens": 4000},
                        {"id": "messages", "tokens": 11000},
                        {"id": "runtime_context", "tokens": 2000},
                        {"id": "other", "tokens": 500}
                    ],
                    "items": [
                        {"category": "system_prompt", "key": "system_prompt", "tokens": 2000},
                        {"category": "system_tools", "key": "API tool schemas", "tokens": 9500},
                        {"category": "memory_files", "key": "repository_instructions", "tokens": 3000},
                        {"category": "skills", "key": "skills", "tokens": 4000},
                        {"category": "messages", "key": "Conversation text", "tokens": 11000},
                        {"category": "runtime_context", "key": "execution_context", "tokens": 2000},
                        {"category": "other", "key": "Additional context", "tokens": 500}
                    ]
                }}});
                update["_meta"]["aster"]["context"]["contentItems"] = json!([
                    "system_prompt",
                    "system_tools",
                    "memory_files",
                    "skills",
                    "messages",
                    "runtime_context",
                    "other"
                ]
                .into_iter()
                .flat_map(|category| {
                    preview_context_sections(category).into_iter().map(move |section| json!({
                        "category": category, "key": section["key"], "chars": section["characters"]
                    }))
                })
                .collect::<Vec<_>>());
            }
            "usage-unknown-window" => update["size"] = json!(0),
            "usage-no-price" => update["cost"] = Value::Null,
            "usage-unavailable" => {
                update["used"] = json!(0);
                update["size"] = json!(0);
                update["cost"] = Value::Null;
                update["_meta"] = json!({"aster": {"usageAvailable": false}});
            }
            "usage-free" => update["cost"]["amount"] = json!(0),
            _ => {}
        }
        if mode.starts_with("hud") {
            let now = chrono::Local::now().timestamp();
            let day = chrono::Local::now().format("%Y-%m-%d").to_string();
            update["cost"]["amount"] = json!(0.0862);
            update["_meta"] = json!({"aster":{"hud":{
                "streamId":"a".repeat(32), "sequence":12, "calls":12, "restoredCalls":3,
                "failedCalls":1, "startedAt":now-1200,
                "tokens":{"input":320000,"output":12400,"cacheRead":256000,"cacheWrite":18000,"reasoning":2100},
                "days":{day:{"cost":0.0862,"complete":true,"calls":9}},
                "latest":{"model":"deepseek-v4.1-flash","respondedAt":now-3,"duration":6.2,"speed":38.4,
                    "tokens":{"input":18000,"output":238,"cacheRead":17000,"cacheWrite":0,"reasoning":90},
                    "limits":[{"label":"tokens","percent":56,"resetsAt":now+1800}],
                    "cacheExpiresAt":now+270},
                "environment":{"skills":["aster.repo","aster.shell","review"],"availableSkills":12,
                    "mcpServers":["docs"],"effort":"high"}
            }}});
            if mode == "hud-unknown" {
                update["used"] = json!(0);
                update["size"] = json!(0);
                update["cost"] = Value::Null;
                update["_meta"]["aster"]["usageAvailable"] = json!(false);
                update["_meta"]["aster"]["hud"]["tokens"] = json!({});
                update["_meta"]["aster"]["hud"]["latest"] = json!({});
            }
        }
        // Use the production notification parser, not direct display strings.
        let mut controller = Controller::new(state, None, false);
        controller.message(json!({
            "method": "session/update",
            "params": {"sessionId": "session-current", "update": update}
        }));
        if let Some((category, selection)) = match mode {
            "context-system" => Some(("system_prompt", 0)),
            "context-messages" => Some(("messages", 4)),
            "context-output" => Some(("output", 7)),
            _ => None,
        } {
            controller.input("/context".into());
            for _ in 0..selection {
                controller.key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE));
            }
            let actions = controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
            let Action::Request {
                method,
                params,
                pending,
            } = actions.into_iter().next().unwrap()
            else {
                panic!("expected a context inspection request");
            };
            assert_eq!(method, "_aster/session/context");
            assert_eq!(params["category"], category);
            controller.registered(42, pending);
            assert!(controller.message(json!({
                "id": 42, "result": {
                    "requestId": params["requestId"], "category": category, "key": null,
                    "source": if category == "output" { "model_response" } else { "model_request" },
                    "sections": preview_context_sections(category)
                }
            })).is_empty());
        }
        state = controller.state;
    }
    if mode.starts_with("hud") {
        state.runtime_supported = true;
        state.approval_policy = "ask".into();
        state.runtime = json!({"reasoningLevel":"high"});
        state.hud.host.git = Some(aster_tui::workspace_changes::GitSummary {
            branch: "main".into(),
            dirty: true,
            ahead: 1,
            modified: 3,
            added: 1,
            untracked: 1,
            lines_added: 85,
            lines_deleted: 12,
            line_counts_available: true,
            ..Default::default()
        });
        state.hud.host.memory = Some((6_442_450_944, 34_359_738_368));
        state.hud.host.instruction_files = vec!["workspace/aster/AGENTS.md".into()];
        if mode != "hud-unknown" {
            state.hud.host.daily_cost = Some(1.24);
            state.hud.host.weekly_cost = Some(8.79);
        }
        state.tasks = json!({
            "steps":[{"title":"Read HUD sources","status":"completed"},
                     {"title":"Wire native telemetry","status":"completed"},
                     {"title":"Check narrow layouts","status":"in_progress"}],
            "currentTask":"Check narrow layouts",
            "jobs":[{"title":"Review TUI layout","status":"running","elapsedSeconds":65},
                    {"title":"Check ACP usage","status":"done","elapsedSeconds":23}]
        });
        match mode {
            "hud-full" | "hud-tokens" | "hud-unknown" | "hud-monochrome" => {
                state.hud.configure("full").unwrap()
            }
            "hud-compact" => state.hud.configure("compact").unwrap(),
            "hud-zh" => {
                state.hud.configure("full").unwrap();
                state.hud.configure("set language zh").unwrap();
            }
            "hud-settings" => state.open(Overlay::Hud),
            _ => {}
        }
        if mode == "hud-tokens" {
            state.hud.configure("set context_value tokens").unwrap();
        }
        state.monochrome = mode == "hud-monochrome";
    }
    if matches!(mode, "welcome-hud-full" | "sent-hud-full") {
        state.hud.configure("full").unwrap();
    }
    if mode == "welcome-hud-full" {
        state.context_window = Some(1_048_576);
        state.hud.config.context_value = "tokens".into();
    }
    if mode == "sent-hud-full" {
        state.session_id = Some("session-current".into());
        let mut controller = Controller::new(state, None, false);
        controller.paste("检查发消息后的状态栏".into());
        let actions = controller.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(
            actions.as_slice(),
            [Action::Request {
                method: "session/prompt",
                ..
            }]
        ));
        state = controller.state;
    }
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    let mut hits = ui::HitAreas::default();
    let mut buffer = terminal
        .draw(|frame| {
            hits = ui::draw(frame, &mut state);
        })
        .unwrap()
        .buffer
        .clone();
    if let Some(request) = state.terminal_image.take_request() {
        assert!(state.terminal_image.complete(request.render()));
        buffer = terminal
            .draw(|frame| hits = ui::draw(frame, &mut state))
            .unwrap()
            .buffer
            .clone();
    }
    let mut copied_selection = None;
    if selecting_text {
        let context = mode.starts_with("context-");
        let composer = mode == "selection-compose";
        let area = if context {
            hits.context_content
                .expect("selected context text viewport")
        } else if composer {
            hits.composer_text
        } else if state.overlay.is_some() {
            hits.text_regions
                .iter()
                .find(|region| region.id == 2)
                .expect("selected detail viewport")
                .area
        } else {
            hits.transcript
        };
        let find = |needle: &str, after: bool| {
            for y in area.y..area.bottom() {
                let mut text = String::new();
                let mut continuation = 0;
                for x in area.x..area.right() {
                    if continuation > 0 {
                        continuation -= 1;
                        continue;
                    }
                    let symbol = buffer[(x, y)].symbol();
                    text.push_str(symbol);
                    continuation = symbol.width().saturating_sub(1);
                }
                if let Some(byte) = text.find(needle) {
                    return (
                        area.x
                            + text[..byte].width() as u16
                            + if after { needle.width() as u16 } else { 0 },
                        y,
                    );
                }
            }
            panic!("selection preview missing {needle:?}");
        };
        let (first, last) = match mode {
            "context-system" => ("You are Aster,", "run checks."),
            "context-messages" => ("User:", "42"),
            "context-output" => ("\"message\":", "查看正文已接上。"),
            "selection-chat" => ("Drag to copy", "中文 and 👩‍💻."),
            "selection-tool" | "selection-tools" => ("Copied tool output:", "second line kept"),
            "selection-compose" => ("拖选就复制", "保留这行草稿"),
            "selection-help" => ("Send / queue", "while working"),
            "selection-diagnostics" => ("Diagnostic output:", "second line"),
            _ => panic!("selection preview requires a text fixture"),
        };
        let start = find(first, false);
        let end = find(last, true);
        let mut controller = Controller::new(state, None, false);
        for (kind, at) in [
            (MouseEventKind::Down(MouseButton::Left), start),
            (MouseEventKind::Drag(MouseButton::Left), end),
            (MouseEventKind::Up(MouseButton::Left), end),
        ] {
            let mouse = MouseEvent {
                kind,
                column: at.0,
                row: at.1,
                modifiers: KeyModifiers::NONE,
            };
            let actions = if context {
                controller
                    .context_mouse(Some(area), mouse)
                    .expect("context pointer was handled")
            } else if composer {
                assert!(controller.composer_mouse(hits.composer_text, mouse));
                if kind == MouseEventKind::Up(MouseButton::Left) {
                    controller.copy_composer_selection()
                } else {
                    vec![]
                }
            } else {
                let outcome = controller.reading_mouse(&hits.text_regions, mouse);
                assert!(outcome.handled && outcome.click.is_none());
                outcome
                    .copy
                    .map(|text| vec![Action::CopySelection { text, cut: false }])
                    .unwrap_or_default()
            };
            if kind == MouseEventKind::Up(MouseButton::Left) {
                let [Action::CopySelection { text, cut: false }] = actions.as_slice() else {
                    panic!("expected selection auto-copy");
                };
                assert!(text.starts_with(first) && text.ends_with(last));
                copied_selection = Some(text.clone());
                // Model a successful writer completion without touching the clipboard.
                controller.state.confirm_copy(text.chars().count());
            } else {
                assert!(actions.is_empty());
            }
            terminal
                .draw(|frame| hits = ui::draw(frame, &mut controller.state))
                .unwrap();
        }
        state = controller.state;
        buffer = terminal
            .draw(|frame| hits = ui::draw(frame, &mut state))
            .unwrap()
            .buffer
            .clone();
    }
    let native_images: Vec<_> = hits
        .context_grid
        .into_iter()
        .map(|area| {
            json!({
                "x": area.x, "y": area.y, "columns": area.width, "rows": area.height,
                "sixel": buffer.cell((area.x, area.y)).unwrap().symbol(),
            })
        })
        .collect();
    let mut cells = Vec::new();
    for row in buffer.content.chunks(width as usize) {
        let mut continuation = 0;
        let mut previous_fg = String::new();
        let mut previous_bg = String::new();
        for cell in row {
            if cell.diff_option == ratatui::buffer::CellDiffOption::Skip
                || cell.symbol().contains('\x1b')
            {
                cells.push(json!({"text":"", "fg":color(cell.fg, "#e1e1e1"),
                    "bg":color(cell.bg, "#141414"), "bold":false}));
                continuation = 0;
                continue;
            }
            if continuation > 0 {
                cells.push(json!({"text":"", "fg":previous_fg, "bg":previous_bg, "bold":false}));
                continuation -= 1;
                continue;
            }
            let (foreground, background) =
                if cell.modifier.contains(ratatui::style::Modifier::REVERSED) {
                    (cell.bg, cell.fg)
                } else {
                    (cell.fg, cell.bg)
                };
            previous_fg = color(foreground, "#e1e1e1");
            previous_bg = color(background, "#141414");
            continuation = cell.symbol().width().saturating_sub(1);
            cells.push(
                json!({"text":cell.symbol(),"fg":previous_fg,"bg":previous_bg,
                "bold":cell.modifier.contains(ratatui::style::Modifier::BOLD)}),
            );
        }
    }
    println!(
        "{}",
        json!({
            "width":width,"height":height,"cells":cells,
            "nativeImages":native_images,"cellPixels":{"width":10,"height":23},
            "source":"offline preview", "mode":requested_mode,
            "selectionCopy":copied_selection,
            "composer":{"x":hits.composer.x,"y":hits.composer.y,
                "width":hits.composer.width,"height":hits.composer.height},
            "hud":hits.hud_rows.iter().map(|(area, command)| json!({"x":area.x,"y":area.y,"width":area.width,"command":command})).collect::<Vec<_>>(),
            "usageSource":if has_sample_usage { "synthetic ACP notification" } else { "none" }
            ,"contentSource":if matches!(mode, "context-system" | "context-messages" | "context-output") {
                "synthetic context RPC response"
            } else { "none" }
        })
    );
}
