// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use tui_textarea::TextArea;

pub(crate) const PYTHON_SANDBOX_SCOPE: &str =
    "Shell/file tools stay on host. Python objects are not migrated; saved state stays.";

#[derive(Clone, Debug)]
pub struct Tool {
    pub id: String,
    pub title: String,
    pub status: String,
    pub content: String,
    pub diff: String,
    pub expanded: bool,
    pub started: Option<Instant>,
    pub elapsed: Option<f64>,
}

impl Tool {
    pub fn elapsed_seconds(&self) -> Option<f64> {
        self.elapsed.or_else(|| {
            if matches!(self.status.as_str(), "pending" | "in_progress") {
                self.started.map(|start| start.elapsed().as_secs_f64())
            } else {
                None
            }
        })
    }
}

#[derive(Clone, Debug)]
pub enum Entry {
    User(String),
    Assistant(String),
    Tool(Tool),
    Notice { text: String, error: bool },
}

impl Entry {
    pub(crate) fn user(draft: &crate::drafts::MessageDraft) -> Self {
        let text = clean(&draft.text);
        if draft.attachments.is_empty() {
            return Self::User(text);
        }
        let mut lines = Vec::new();
        for attachment in &draft.attachments {
            let bytes = attachment.byte_len();
            let size = if bytes < 1024 {
                format!("{bytes} B")
            } else if bytes < 1024 * 1024 {
                format!("{:.1} KiB", bytes as f64 / 1024.0)
            } else {
                format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
            };
            let kind = if attachment.is_image() {
                "Image"
            } else {
                "File"
            };
            let format = if attachment.is_image() {
                attachment
                    .mime_type()
                    .trim_start_matches("image/")
                    .to_uppercase()
            } else {
                attachment.mime_type().into()
            };
            lines.push(format!(
                "[{kind}] {} · {format} · {size}",
                clean(attachment.name()).replace(['\n', '\r', '\t'], " ")
            ));
        }
        if !text.trim().is_empty() {
            lines.push(text);
        }
        Self::User(lines.join("\n"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyConfirmation {
    pub characters: usize,
    pub expires_at: Instant,
}

#[derive(Clone, Debug, Default)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub updated: String,
}

#[derive(Clone, Debug, Default)]
pub struct Command {
    pub name: String,
    pub description: String,
    pub argument_hint: String,
    pub completions: Vec<String>,
    pub is_skill: bool,
    pub aliases: Vec<String>,
}

impl Command {
    fn label(&self, name: &str) -> String {
        if self.argument_hint.is_empty() {
            name.into()
        } else {
            format!("{name} {}", self.argument_hint)
        }
    }

    fn matches_name(&self, name: &str) -> bool {
        self.name == name
            || (self.is_skill && {
                let query = name.to_lowercase();
                self.name.to_lowercase() == query
                    || self
                        .aliases
                        .iter()
                        .any(|alias| alias.to_lowercase() == query)
            })
    }

    fn completion_name(&self, query: &str) -> Option<&str> {
        let mut names =
            std::iter::once(self.name.as_str()).chain(self.aliases.iter().map(String::as_str));
        names
            .clone()
            .find(|name| name.to_lowercase() == query)
            .or_else(|| names.find(|name| name.to_lowercase().starts_with(query)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Connecting,
    Preparing,
    Ready,
    Busy,
    Cancelling,
    Closing,
    Failed,
}

#[derive(Default)]
pub struct ModelSelection {
    pub supported: bool,
    pub loading: bool,
    pub error: String,
    pub generation: u64,
}

#[derive(Default)]
pub struct SkillsBrowser {
    pub supported: bool,
    pub loading: bool,
    pub error: Option<String>,
    pub items: Vec<Value>,
    pub detail: Option<Value>,
    pub generation: u64,
    pub total: usize,
    pub truncated: bool,
}

#[derive(Default)]
pub struct McpBrowser {
    pub supported: bool,
    pub loading: bool,
    pub error: Option<String>,
    pub servers: Vec<Value>,
    pub warnings: Vec<String>,
    pub generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlay {
    Commands,
    Sessions,
    Drafts,
    Tools,
    Changes,
    Help,
    Diagnostics,
    Hud,
    HudInfo,
    Skills,
    Mcp,
    Context,
    Queue,
    Search,
    Rewind,
    Tree,
    Files,
    Attachments,
    Models,
    Reasoning,
    Modes,
    Permissions,
    Sandbox,
    Approval,
    Tasks,
    Rollback,
    RollbackConfirm,
    UndoConfirm,
}

#[derive(Clone, Debug)]
pub struct RewindPoint {
    pub user_message_index: usize,
    pub prompt: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SteeringState {
    Sending,
    Accepted,
    Applied,
    Cancelled,
    Interrupted,
    TurnFinished,
    Rejected,
}

impl SteeringState {
    pub fn from_wire(value: &Value) -> Option<Self> {
        // The protocol uses string states. Serde also accepts externally tagged
        // objects for unit variants, which are not valid steering receipts.
        value.as_str()?;
        serde_json::from_value(value.clone()).ok()
    }

    pub fn is_pending(self) -> bool {
        match self {
            Self::Sending | Self::Accepted => true,
            Self::Applied
            | Self::Cancelled
            | Self::Interrupted
            | Self::TurnFinished
            | Self::Rejected => false,
        }
    }

    pub fn is_terminal(self) -> bool {
        !self.is_pending()
    }

    pub fn is_acknowledged(self) -> bool {
        matches!(self, Self::Accepted | Self::Applied)
    }

    /// Notifications and replies may arrive in either order. A terminal receipt
    /// cannot become pending again; an applied receipt cannot be downgraded.
    pub fn can_transition_to(self, next: Self) -> bool {
        match self {
            Self::Applied => next == Self::Applied,
            _ if self.is_terminal() => next.is_terminal(),
            _ => true,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Sending => "Sending correction…",
            Self::Accepted => "Correction received · waiting for the next model request",
            Self::Applied => "Correction included in the model request",
            Self::Cancelled => "Correction not applied · task cancelled",
            Self::Interrupted => "Correction interrupted · review before resending",
            Self::TurnFinished => "Correction not applied · task already finished",
            Self::Rejected => "Correction rejected · draft kept",
        }
    }

    pub fn is_error(self) -> bool {
        matches!(
            self,
            Self::Rejected | Self::Interrupted | Self::TurnFinished
        )
    }
}

#[derive(Clone, Debug)]
pub struct SteeringMessage {
    pub id: String,
    pub turn_id: String,
    pub text: String,
    pub state: SteeringState,
    pub reason: String,
    pub entry: usize,
}

#[derive(Clone, Debug)]
pub struct Choice {
    pub label: String,
    pub detail: String,
    pub value: String,
}

pub(crate) struct DraftCompletion {
    pub row: usize,
    pub start: usize,
    pub end: usize,
    pub choices: Vec<Choice>,
}

#[derive(Clone, Debug)]
pub struct ApprovalOption {
    pub id: String,
    pub name: String,
    pub kind: String,
}

#[derive(Clone, Debug)]
pub struct ApprovalPreview {
    pub summary: String,
    pub details: Vec<String>,
    /// Display completeness only; this never changes execution permissions.
    pub complete: bool,
}

#[derive(Clone, Debug)]
pub struct ApprovalRequest {
    pub id: Value,
    pub title: String,
    pub code: String,
    pub preview: Option<ApprovalPreview>,
    pub show_code: bool,
    pub options: Vec<ApprovalOption>,
}

impl ApprovalRequest {
    pub fn default_selection(&self) -> usize {
        self.options
            .iter()
            .position(|option| option.kind == "allow_once")
            .or_else(|| {
                self.options
                    .iter()
                    .position(|option| option.kind.starts_with("reject"))
            })
            .unwrap_or(0)
    }
}

#[derive(Default)]
pub struct RollbackView {
    pub file: Option<crate::workspace_changes::ChangedFile>,
    pub hunks: Vec<String>,
    pub plan: Option<crate::rollback::RollbackPlan>,
    pub receipt: Option<crate::rollback::RollbackReceipt>,
    pub loading: bool,
}

pub(crate) struct RemovedAttachment {
    pub attachment: crate::attachments::Attachment,
    pub index: usize,
    pub generation: u64,
}

pub struct State {
    pub workspace: String,
    pub model: String,
    pub sandbox: bool,
    pub session_id: Option<String>,
    /// Changes on a conversation switch, not when background preparation returns its ID.
    pub conversation_generation: u64,
    pub title: String,
    pub phase: Phase,
    pub(crate) sandbox_preparing: bool,
    pub status: String,
    /// A second Ctrl+C before this instant confirms the user's exit request.
    pub exit_confirmation: Option<Instant>,
    /// Only an idle, unconsumed Esc press can start the double-Esc gesture.
    pub escape_confirmation: Option<Instant>,
    /// Brief success feedback after the clipboard writer completes.
    pub copy_confirmation: Option<CopyConfirmation>,
    pub rewind_points: Vec<RewindPoint>,
    pub rewind_loading: bool,
    pub session_tree: crate::session_tree::SessionTree,
    pub tree_loading: bool,
    pub tree_search_supported: bool,
    pub tree_search_error: Option<String>,
    pub steering_supported: bool,
    pub active_turn_id: Option<String>,
    pub steering_messages: Vec<SteeringMessage>,
    pub entries: Vec<Entry>,
    /// Full transcript details explicitly opened by the user or a search jump.
    pub expanded_entries: HashSet<usize>,
    pub sessions: Vec<Session>,
    pub saved_drafts: Vec<crate::drafts::LoadedDraft>,
    pub commands: Vec<Command>,
    pub commands_received: bool,
    pub editor: TextArea<'static>,
    pub(crate) composer: crate::composer::ComposerState,
    pub(crate) reading_selection: crate::text_selection::SelectionState,
    pub(crate) reading_cache: crate::text_selection::Cache,
    /// Invalidates asynchronous edits even when a later draft has identical text.
    pub draft_revision: u64,
    pub overlay: Option<Overlay>,
    pub query: String,
    pub selected: usize,
    pub slash_selected: usize,
    pub slash_dismissed: bool,
    /// Unsent messages belong to the current conversation and remain in FIFO order.
    pub queued: VecDeque<crate::drafts::MessageDraft>,
    pub attachments: Vec<crate::attachments::Attachment>,
    pub(crate) removed_attachment: Option<RemovedAttachment>,
    pub(crate) image_preview: crate::image_preview::PreviewCache,
    pub terminal_graphics: crate::terminal_input::GraphicsCapabilities,
    pub terminal_image: crate::terminal_image::PreviewCache,
    pub context_grid: crate::context::GridRasterCache,
    /// Image-viewer status remains visible while the attachment inspector is open.
    pub original_image_notice: Option<(String, bool)>,
    pub attachments_supported: bool,
    pub image_input_supported: bool,
    pub attachment_loading: bool,
    pub file_candidates: Vec<String>,
    pub files_loading: bool,
    pub files_notice: String,
    pub recovery: Option<crate::drafts::MessageDraft>,
    pub runtime_supported: bool,
    pub runtime: Value,
    pub model_selection: ModelSelection,
    pub skills: SkillsBrowser,
    pub mcp: McpBrowser,
    /// Effort changes stay local to the model picker until Enter confirms them.
    pub model_effort_drafts: HashMap<String, Option<String>>,
    pub mode: String,
    pub approval_policy: String,
    pub approvals: VecDeque<ApprovalRequest>,
    pub tasks_supported: bool,
    pub tasks: Value,
    pub queue_paused: bool,
    /// The next render should reveal this transcript entry after a search jump.
    pub focus_entry: Option<usize>,
    pub search_highlight: String,
    /// Number of wrapped transcript rows above the newest output.
    pub scroll: usize,
    pub(crate) rendered_rows: Option<(usize, u16)>,
    pub(crate) transcript_cache: crate::ui::TranscriptCache,
    pub detail_scroll: u16,
    /// Actual approval viewport rows, so paging cannot skip unreviewed code.
    pub(crate) approval_page_rows: u16,
    pub show_sidebar: bool,
    pub usage: String,
    pub hud: crate::hud::Hud,
    /// Known model capacity, independent of a reported context measurement.
    pub context_window: Option<u64>,
    pub context_ratio: Option<f64>,
    pub context_snapshot: Option<crate::context::ContextSnapshot>,
    pub context_expanded: bool,
    pub context_view: Option<crate::context_detail::View>,
    pub context_view_generation: u64,
    pub context_reveal_selection: bool,
    pub cost: String,
    pub started: Option<Instant>,
    pub last_elapsed: Option<f64>,
    pub diagnostics: VecDeque<String>,
    pub workspace_changes: Option<crate::workspace_changes::WorkspaceChanges>,
    pub changes_loading: bool,
    pub changes_error: Option<String>,
    pub conversation_changes: bool,
    pub rollback: RollbackView,
    pub monochrome: bool,
}

impl State {
    pub fn new(workspace: String, model: String, sandbox: bool) -> Self {
        Self {
            workspace,
            model,
            sandbox,
            session_id: None,
            conversation_generation: 0,
            title: String::new(),
            phase: Phase::Connecting,
            sandbox_preparing: false,
            status: "Starting agent…".into(),
            exit_confirmation: None,
            escape_confirmation: None,
            copy_confirmation: None,
            rewind_points: Vec::new(),
            rewind_loading: false,
            session_tree: crate::session_tree::SessionTree::default(),
            tree_loading: false,
            tree_search_supported: false,
            tree_search_error: None,
            steering_supported: false,
            active_turn_id: None,
            steering_messages: Vec::new(),
            entries: Vec::new(),
            expanded_entries: HashSet::new(),
            sessions: Vec::new(),
            saved_drafts: Vec::new(),
            commands: default_commands(),
            commands_received: false,
            editor: TextArea::default(),
            composer: crate::composer::ComposerState::default(),
            reading_selection: crate::text_selection::SelectionState::default(),
            reading_cache: crate::text_selection::Cache::default(),
            draft_revision: 0,
            overlay: None,
            query: String::new(),
            selected: 0,
            slash_selected: 0,
            slash_dismissed: false,
            queued: VecDeque::new(),
            attachments: Vec::new(),
            removed_attachment: None,
            image_preview: crate::image_preview::PreviewCache::default(),
            terminal_graphics: crate::terminal_input::GraphicsCapabilities::default(),
            terminal_image: crate::terminal_image::PreviewCache::default(),
            context_grid: crate::context::GridRasterCache::default(),
            original_image_notice: None,
            attachments_supported: false,
            image_input_supported: false,
            attachment_loading: false,
            file_candidates: Vec::new(),
            files_loading: false,
            files_notice: String::new(),
            recovery: None,
            runtime_supported: false,
            runtime: Value::Null,
            model_selection: ModelSelection::default(),
            skills: SkillsBrowser::default(),
            mcp: McpBrowser::default(),
            model_effort_drafts: HashMap::new(),
            mode: "code".into(),
            approval_policy: "backend".into(),
            approvals: VecDeque::new(),
            tasks_supported: false,
            tasks: Value::Null,
            queue_paused: false,
            focus_entry: None,
            search_highlight: String::new(),
            scroll: 0,
            rendered_rows: None,
            transcript_cache: crate::ui::TranscriptCache::default(),
            detail_scroll: 0,
            approval_page_rows: 1,
            show_sidebar: false,
            usage: String::new(),
            hud: crate::hud::Hud::default(),
            context_window: None,
            context_ratio: None,
            context_snapshot: None,
            context_expanded: false,
            context_view: None,
            context_view_generation: 0,
            context_reveal_selection: false,
            cost: String::new(),
            started: None,
            last_elapsed: None,
            diagnostics: VecDeque::new(),
            workspace_changes: None,
            changes_loading: false,
            changes_error: None,
            conversation_changes: false,
            rollback: RollbackView::default(),
            monochrome: std::env::var_os("NO_COLOR").is_some(),
        }
    }

    pub fn busy(&self) -> bool {
        matches!(
            self.phase,
            Phase::Connecting | Phase::Preparing | Phase::Busy | Phase::Cancelling | Phase::Closing
        )
    }

    pub fn preparing(&self) -> bool {
        matches!(self.phase, Phase::Connecting | Phase::Preparing)
    }

    pub fn can_cancel(&self) -> bool {
        self.phase == Phase::Busy && (self.active_turn_id.is_some() || self.sandbox_preparing)
    }

    pub fn can_undo_attachment(&self) -> bool {
        self.removed_attachment
            .as_ref()
            .is_some_and(|removed| removed.generation == self.conversation_generation)
    }

    pub(crate) fn draft_changed(&mut self) {
        self.reading_selection.clear();
        self.draft_revision = self.draft_revision.wrapping_add(1);
        self.removed_attachment = None;
        self.composer.reset_navigation();
    }

    pub(crate) fn overlay_searchable(&self) -> bool {
        matches!(
            self.overlay,
            Some(
                Overlay::Commands
                    | Overlay::Sessions
                    | Overlay::Drafts
                    | Overlay::Tools
                    | Overlay::Search
                    | Overlay::Queue
                    | Overlay::Rewind
                    | Overlay::Tree
                    | Overlay::Files
                    | Overlay::Attachments
                    | Overlay::Models
                    | Overlay::Modes
                    | Overlay::Permissions
                    | Overlay::Sandbox
                    | Overlay::Tasks
                    | Overlay::Mcp
                    | Overlay::Hud
            )
        ) || (self.overlay == Some(Overlay::Skills) && self.skills.detail.is_none())
            || (self.overlay == Some(Overlay::Changes) && !self.conversation_changes)
    }

    pub fn open(&mut self, overlay: Overlay) {
        self.reading_selection.clear();
        self.reading_cache.clear();
        self.escape_confirmation = None;
        self.original_image_notice = None;
        self.composer.cancel_pointer();
        self.overlay = Some(overlay);
        self.query.clear();
        self.selected = 0;
        self.detail_scroll = 0;
        self.approval_page_rows = 1;
        self.context_view = None;
        self.context_view_generation = self.context_view_generation.wrapping_add(1);
        self.context_reveal_selection = false;
        self.model_effort_drafts.clear();
        self.skills.generation = self.skills.generation.wrapping_add(1);
        self.skills.detail = None;
        self.skills.loading = false;
        self.mcp.generation = self.mcp.generation.wrapping_add(1);
        self.mcp.loading = false;
        self.focus_runtime_choice();
    }

    pub(crate) fn focus_runtime_choice(&mut self) {
        let value = match self.overlay {
            Some(Overlay::Models) => self.model.as_str(),
            Some(Overlay::Reasoning) => self.runtime["reasoningSelection"].as_str().unwrap_or(""),
            Some(Overlay::Modes) => self.mode.as_str(),
            Some(Overlay::Permissions) => self.permission_mode(),
            Some(Overlay::Sandbox) => self.runtime["executionBackend"].as_str().unwrap_or(""),
            _ => return,
        };
        self.selected = self
            .choices()
            .iter()
            .position(|choice| choice.value == value)
            .unwrap_or(0);
    }

    pub fn permission_mode(&self) -> &str {
        match self.runtime["permissionMode"].as_str() {
            Some(mode @ ("auto" | "manual" | "accept_edits" | "plan")) => mode,
            _ if self.mode == "plan" => "plan",
            _ => match self.approval_policy.as_str() {
                "ask" | "manual" => "manual",
                "session" | "auto" => "auto",
                other => other,
            },
        }
    }

    pub fn permission_label(&self) -> &str {
        match self.permission_mode() {
            "auto" => "Auto",
            "manual" => "Manual",
            "accept_edits" => "Accept edits",
            "plan" => "Plan",
            "backend" => "Backend default",
            other => other,
        }
    }

    fn model_reasoning_metadata(&self, model: &str) -> Option<&Value> {
        if model == self.model {
            Some(&self.runtime)
        } else {
            self.runtime["models"]
                .as_array()?
                .iter()
                .find(|entry| entry["id"].as_str() == Some(model))
        }
    }

    pub(crate) fn model_effort_choices(&self, model: &str) -> (Vec<Choice>, usize) {
        let Some(metadata) = self.model_reasoning_metadata(model) else {
            return (Vec::new(), 0);
        };
        let choices = reasoning_choices(metadata);
        let selection = self.model_effort_drafts.get(model).map_or_else(
            || metadata["reasoningSelection"].as_str().unwrap_or(""),
            |level| level.as_deref().unwrap_or(""),
        );
        let selected = choices
            .iter()
            .position(|choice| choice.value == selection)
            .unwrap_or(0);
        (choices, selected)
    }

    pub(crate) fn adjust_model_effort(&mut self, increase: bool) {
        if !self.runtime_supported || self.session_id.is_none() || self.model.is_empty() {
            return;
        }
        let Some(model) = self
            .choices()
            .get(self.selected)
            .map(|choice| choice.value.clone())
        else {
            return;
        };
        let (levels, current) = self.model_effort_choices(&model);
        if levels.len() < 2 {
            return;
        }
        let next = if increase {
            (current + 1).min(levels.len() - 1)
        } else {
            current.saturating_sub(1)
        };
        if next != current {
            let value = levels[next].value.clone();
            self.model_effort_drafts
                .insert(model, (!value.is_empty()).then_some(value));
        }
    }

    pub(crate) fn text_scope(&self) -> crate::text_selection::Scope {
        let subject = if self.overlay.is_some() {
            if let Some(view) = self.context_view.as_ref() {
                format!("{}:{}", view.generation, view.request_id)
            } else {
                self.choices()
                    .get(self.selected)
                    .map(|choice| choice.value.clone())
                    .unwrap_or_default()
            }
        } else {
            String::new()
        };
        crate::text_selection::Scope {
            conversation: self.conversation_generation,
            overlay: self.overlay,
            subject,
            query: if self.overlay.is_some() {
                self.query.clone()
            } else {
                String::new()
            },
        }
    }

    pub fn notice(&mut self, text: impl Into<String>, error: bool) {
        self.entries.push(Entry::Notice {
            text: clean(&text.into()),
            error,
        });
    }

    pub fn confirm_copy(&mut self, characters: usize) {
        self.copy_confirmation = Some(CopyConfirmation {
            characters,
            expires_at: Instant::now() + Duration::from_secs(2),
        });
    }

    pub fn tree_search_pending(&self) -> bool {
        self.tree_search_supported
            && !self.query.trim().is_empty()
            && self.tree_search_error.is_none()
            && !self.session_tree.has_search_results(&self.query)
    }

    pub fn is_skill_command(&self, name: &str) -> bool {
        self.commands
            .iter()
            .find(|command| command.matches_name(name))
            .is_some_and(|command| command.is_skill)
    }

    pub fn is_known_command(&self, name: &str) -> bool {
        matches!(name, "/" | "/quit")
            || self
                .commands
                .iter()
                .any(|command| command.matches_name(name))
    }

    pub fn slash_choice_fills_draft(&self, choice: &Choice) -> bool {
        choice.value.starts_with('$')
            || choice.value.chars().any(char::is_whitespace)
            || self.commands.iter().any(|command| {
                command.matches_name(&choice.value)
                    && (command.is_skill || !command.argument_hint.is_empty())
            })
    }

    pub fn slash_choice_is_exact(&self, choice: &Choice) -> bool {
        let command = self.editor.lines()[0]
            .split_whitespace()
            .next()
            .unwrap_or("");
        command == "/" || choice.value.eq_ignore_ascii_case(command)
    }

    /// Slash commands and inline skill mentions share the composer menu.
    pub fn slash_choices(&self) -> Vec<Choice> {
        self.draft_completion()
            .map_or_else(Vec::new, |completion| completion.choices)
    }

    pub(crate) fn draft_completion(&self) -> Option<DraftCompletion> {
        if self.overlay.is_some()
            || self.slash_dismissed
            || self
                .editor
                .selection_range()
                .is_some_and(|(start, end)| start != end)
        {
            return None;
        }
        let tui_textarea::DataCursor(row, column) = self.editor.cursor();
        let line = self.editor.lines().get(row)?;
        let characters: Vec<_> = line.chars().collect();

        let mut start = column;
        while start > 0 && skill_name_character(characters[start - 1]) {
            start -= 1;
        }
        if start > 0 && characters[start - 1] == '$' {
            let marker = start - 1;
            if marker > 0
                && (skill_name_character(characters[marker - 1])
                    || matches!(characters[marker - 1], '/' | '\\' | '$' | '='))
            {
                return None;
            }
            if skill_reference_in_code(self.editor.lines(), row, marker) {
                return None;
            }
            let mut end = column;
            while end < characters.len() && skill_name_character(characters[end]) {
                end += 1;
            }
            if characters
                .get(end)
                .is_some_and(|value| matches!(value, '/' | '\\'))
            {
                return None;
            }
            let token = format!("/{}", characters[start..end].iter().collect::<String>());
            if !self
                .commands
                .iter()
                .any(|command| command.is_skill && command.matches_name(&token))
            {
                while end > column && characters[end - 1] == '.' {
                    end -= 1;
                }
            }
            let query = format!(
                "/{}",
                characters[start..column]
                    .iter()
                    .collect::<String>()
                    .to_lowercase()
            );
            let choices = self
                .commands
                .iter()
                .filter(|command| command.is_skill)
                .filter_map(|command| {
                    let name = command.completion_name(&query)?.strip_prefix('/')?;
                    Some(Choice {
                        label: format!("${name}"),
                        detail: command.description.clone(),
                        value: format!("${name}"),
                    })
                })
                .collect();
            return Some(DraftCompletion {
                row,
                start: marker,
                end,
                choices,
            });
        }

        if row != 0 {
            return None;
        }
        let start = characters.iter().position(|value| !value.is_whitespace())?;
        if characters[start] != '/' || column <= start {
            return None;
        }
        let end = characters[start..]
            .iter()
            .position(|value| value.is_whitespace())
            .map_or(characters.len(), |length| start + length);
        if column > end {
            // Typed commands can still advertise first-argument completions.
            if self.editor.lines().len() != 1 || column != characters.len() {
                return None;
            }
            let (name, argument) = line.trim_start().split_once(char::is_whitespace)?;
            if argument.chars().any(char::is_whitespace) {
                return None;
            }
            let command = self
                .commands
                .iter()
                .find(|command| command.name.eq_ignore_ascii_case(name))?;
            let query = argument.to_lowercase();
            return Some(DraftCompletion {
                row,
                start,
                end: column,
                choices: command
                    .completions
                    .iter()
                    .filter(|value| value.to_lowercase().starts_with(&query))
                    .map(|value| Choice {
                        label: format!("{} {value}", command.name),
                        detail: "Insert argument; finish your message before sending".into(),
                        value: format!("{} {value}", command.name),
                    })
                    .collect(),
            });
        }
        let token = characters[start..end].iter().collect::<String>();
        // The session response can precede its command catalog.
        if !self.commands_received && token != "/" && !self.is_known_command(&token) {
            return None;
        }
        let query = characters[start..column]
            .iter()
            .collect::<String>()
            .to_lowercase();
        let mut choices: Vec<_> = self
            .commands
            .iter()
            .filter_map(|command| {
                let name = command.completion_name(&query)?;
                Some(Choice {
                    label: command.label(name),
                    detail: command.description.clone(),
                    value: name.into(),
                })
            })
            .collect();
        choices.sort_by_key(|choice| choice.value.to_lowercase() != query);
        Some(DraftCompletion {
            row,
            start,
            end,
            choices,
        })
    }

    pub fn choices(&self) -> Vec<Choice> {
        if self.overlay == Some(Overlay::Context) {
            return self
                .context_snapshot
                .as_ref()
                .map_or_else(Vec::new, |snapshot| {
                    snapshot
                        .targets(self.context_expanded)
                        .into_iter()
                        .enumerate()
                        .map(|(index, target)| Choice {
                            label: target.label(),
                            detail: "Enter to view actual content".into(),
                            value: index.to_string(),
                        })
                        .collect()
                });
        }
        if self.overlay == Some(Overlay::Tree) {
            return if self.tree_loading
                || self.tree_search_pending()
                || self.tree_search_error.is_some()
            {
                Vec::new()
            } else {
                self.session_tree.choices(&self.query)
            };
        }
        if self.overlay == Some(Overlay::Search) {
            return crate::search::choices(&self.entries, &self.query);
        }
        if self.overlay == Some(Overlay::Queue) {
            let query = self.query.to_lowercase();
            return self
                .queued
                .iter()
                .enumerate()
                .filter(|(_, text)| text.to_lowercase().contains(&query))
                .map(|(index, text)| Choice {
                    label: clean(
                        text.lines()
                            .find(|line| !line.trim().is_empty())
                            .unwrap_or(""),
                    ),
                    detail: format!(
                        "#{} · {} lines · {} characters · {} attachments",
                        index + 1,
                        text.lines().count(),
                        text.chars().count(),
                        text.attachments.len()
                    ),
                    value: index.to_string(),
                })
                .collect();
        }
        let query = self.query.to_lowercase();
        let choices: Vec<Choice> = match self.overlay {
            Some(Overlay::Mcp) if !self.mcp.loading && self.mcp.error.is_none() => self
                .mcp
                .servers
                .iter()
                .filter_map(|server| {
                    let name = server["name"].as_str()?;
                    Some(Choice {
                        label: format!(
                            "{} · {}",
                            clean(name),
                            clean(server["state"].as_str().unwrap_or("unknown"))
                        ),
                        detail: format!(
                            "{} · {} · {} tools loaded",
                            clean(server["source"].as_str().unwrap_or("unknown source")),
                            clean(server["transport"].as_str().unwrap_or("unknown transport")),
                            server["toolCount"].as_u64().unwrap_or(0)
                        ),
                        value: name.into(),
                    })
                })
                .collect(),
            Some(Overlay::Skills)
                if !self.skills.loading
                    && self.skills.error.is_none()
                    && self.skills.detail.is_none() =>
            {
                self.skills
                    .items
                    .iter()
                    .filter_map(|skill| {
                        let name = skill["name"].as_str()?;
                        let active = skill["active"] == true;
                        let status = if active {
                            "Active"
                        } else if skill["loaded"] == true {
                            "Available"
                        } else {
                            "Not loaded"
                        };
                        Some(Choice {
                            label: format!("{} {}", if active { "✓" } else { "·" }, clean(name)),
                            detail: format!(
                                "{status} · {}",
                                clean(skill["description"].as_str().unwrap_or(""))
                            ),
                            value: name.into(),
                        })
                    })
                    .collect()
            }
            Some(Overlay::Hud) => self.hud.choices(),
            Some(Overlay::Rollback) => {
                let mut scopes = vec![Choice {
                    label: "All unstaged hunks in this file".into(),
                    detail: "Review the complete rollback before applying".into(),
                    value: "all".into(),
                }];
                scopes.extend(
                    self.rollback
                        .hunks
                        .iter()
                        .enumerate()
                        .map(|(index, header)| Choice {
                            label: format!("Hunk {} · {}", index + 1, clean(header)),
                            detail: "Keep every other hunk".into(),
                            value: index.to_string(),
                        }),
                );
                scopes
            }
            Some(Overlay::RollbackConfirm | Overlay::UndoConfirm) if !self.rollback.loading => {
                vec![
                    Choice {
                        label: "Cancel".into(),
                        detail: "Keep the current file".into(),
                        value: "cancel".into(),
                    },
                    Choice {
                        label: if self.overlay == Some(Overlay::UndoConfirm) {
                            "Undo the previous rollback"
                        } else {
                            "Apply the reviewed rollback"
                        }
                        .into(),
                        detail: "File and index are checked again before writing".into(),
                        value: "apply".into(),
                    },
                ]
            }
            Some(Overlay::Models) => self.runtime["models"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|model| {
                    let id = model["id"].as_str()?;
                    Some(Choice {
                        label: clean(model["name"].as_str().unwrap_or(id)),
                        detail: if id == self.model {
                            "Current model"
                        } else {
                            "Use this configured model for the session"
                        }
                        .into(),
                        value: id.into(),
                    })
                })
                .collect(),
            Some(Overlay::Reasoning) => reasoning_choices(&self.runtime),
            Some(Overlay::Modes) => self.runtime["modes"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|mode| {
                    let id = mode["id"].as_str()?;
                    Some(Choice {
                        label: clean(mode["name"].as_str().unwrap_or(id)),
                        detail: clean(mode["description"].as_str().unwrap_or("")),
                        value: id.into(),
                    })
                })
                .collect(),
            Some(Overlay::Permissions) => vec![
                Choice {
                    label: "Auto".into(),
                    detail: "Run code and change or delete files without asking".into(),
                    value: "auto".into(),
                },
                Choice {
                    label: "Manual".into(),
                    detail: "Ask before running code or making changes".into(),
                    value: "manual".into(),
                },
                Choice {
                    label: "Accept edits".into(),
                    detail: "Accept file reads and edits; ask before commands and other code"
                        .into(),
                    value: "accept_edits".into(),
                },
                Choice {
                    label: "Plan".into(),
                    detail: "Plan from supplied context without running tools".into(),
                    value: "plan".into(),
                },
            ],
            Some(Overlay::Sandbox) => [
                ("native", "Native", "Run Python on the host"),
                ("sandbox", "Python sandbox", "Run Python cells in a microVM"),
            ]
            .into_iter()
            .filter(|(backend, _, _)| {
                self.runtime["executionBackends"]
                    .as_array()
                    .is_some_and(|backends| backends.iter().any(|value| value == backend))
            })
            .map(|(backend, label, detail)| Choice {
                label: label.into(),
                detail: if self.runtime["executionBackend"] == backend {
                    format!("Current · {detail}")
                } else {
                    detail.into()
                },
                value: backend.into(),
            })
            .collect(),
            Some(Overlay::Approval) => self
                .approvals
                .front()
                .map(|approval| {
                    approval
                        .options
                        .iter()
                        .map(|option| Choice {
                            label: option.name.clone(),
                            detail: option.kind.replace('_', " "),
                            value: option.id.clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            Some(Overlay::Tasks) => {
                let mut choices = Vec::new();
                if let Some(steps) = self.tasks["steps"].as_array() {
                    for step in steps {
                        let id = step["id"].as_str().unwrap_or("");
                        choices.push(Choice {
                            label: clean(step["title"].as_str().unwrap_or("Plan step")),
                            detail: clean(step["status"].as_str().unwrap_or("pending")),
                            value: format!("step:{id}"),
                        });
                    }
                }
                if let Some(jobs) = self.tasks["jobs"].as_array() {
                    for job in jobs {
                        let id = job["id"].as_str().unwrap_or("");
                        choices.push(Choice {
                            label: clean(job["title"].as_str().unwrap_or(id)),
                            detail: format!(
                                "Background · {}{}",
                                clean(job["status"].as_str().unwrap_or("unknown")),
                                if job["canStop"] == true {
                                    " · Ctrl+X stop"
                                } else {
                                    ""
                                }
                            ),
                            value: format!("job:{id}"),
                        });
                    }
                }
                choices
            }
            Some(Overlay::Files) => self
                .file_candidates
                .iter()
                .map(|path| Choice {
                    label: clean(path),
                    detail: "Enter attaches a snapshot of this file".into(),
                    value: path.clone(),
                })
                .collect(),
            Some(Overlay::Attachments) => self
                .attachments
                .iter()
                .enumerate()
                .map(|(index, item)| Choice {
                    label: clean(item.name()),
                    detail: format!(
                        "{} · {} bytes · included on send",
                        item.mime_type(),
                        item.byte_len()
                    ),
                    value: index.to_string(),
                })
                .collect(),
            Some(Overlay::Commands) => self
                .commands
                .iter()
                .map(|c| Choice {
                    label: c.label(&c.name),
                    detail: c.description.clone(),
                    value: c.name.clone(),
                })
                .collect(),
            Some(Overlay::Drafts) => self
                .saved_drafts
                .iter()
                .enumerate()
                .map(|(index, draft)| {
                    let snapshot = &draft.snapshot;
                    let input = std::iter::once(&snapshot.composer)
                        .chain(snapshot.in_flight.iter())
                        .chain(snapshot.queued.iter())
                        .find(|input| !input.is_empty());
                    let label = input
                        .and_then(|input| input.text.lines().find(|line| !line.trim().is_empty()))
                        .map(|line| clean(&line.chars().take(160).collect::<String>()))
                        .unwrap_or_else(|| "Attached context".into());
                    let conversation =
                        if crate::drafts::is_provisional_scope(draft.session_id.as_deref()) {
                            "New conversation"
                        } else {
                            "Saved conversation"
                        };
                    Choice {
                        label,
                        detail: format!(
                            "{conversation} · {} queued{}",
                            snapshot.queued.len(),
                            if snapshot.in_flight.is_some() {
                                " · interrupted input"
                            } else {
                                ""
                            }
                        ),
                        value: index.to_string(),
                    }
                })
                .collect(),
            Some(Overlay::Sessions) => self
                .sessions
                .iter()
                .map(|s| Choice {
                    label: if s.title.is_empty() {
                        "Untitled conversation".into()
                    } else {
                        s.title.clone()
                    },
                    detail: format!("{}  {}", s.id, s.updated),
                    value: s.id.clone(),
                })
                .collect(),
            Some(Overlay::Rewind) => {
                return self
                    .rewind_points
                    .iter()
                    .rev()
                    .filter(|point| clean(&point.prompt).to_lowercase().contains(&query))
                    .map(|point| Choice {
                        label: clean(&point.prompt)
                            .lines()
                            .find(|line| !line.trim().is_empty())
                            .unwrap_or("")
                            .to_owned(),
                        detail: format!(
                            "Message {} · edit and continue from before this message",
                            point.user_message_index + 1
                        ),
                        value: point.user_message_index.to_string(),
                    })
                    .collect()
            }
            Some(Overlay::Tree) => {
                unreachable!("tree choices are handled before query normalization")
            }
            Some(Overlay::Tools) => self
                .entries
                .iter()
                .rev()
                .filter_map(|e| match e {
                    Entry::Tool(t) => Some(Choice {
                        label: t.title.clone(),
                        detail: t.status.clone(),
                        value: t.id.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            Some(Overlay::Changes) if !self.conversation_changes => self
                .workspace_changes
                .as_ref()
                .map(|snapshot| {
                    snapshot
                        .files
                        .iter()
                        .enumerate()
                        .map(|(index, file)| Choice {
                            label: clean(&file.path),
                            detail: format!("{} · {}", file.stage.label(), clean(&file.status)),
                            value: index.to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        choices
            .into_iter()
            .filter(|c| {
                format!("{} {}", c.label, c.detail)
                    .to_lowercase()
                    .contains(&query)
            })
            .collect()
    }

    pub fn selected_tool(&self) -> Option<&Tool> {
        let choices = self.choices();
        let id = &choices.get(self.selected)?.value;
        self.entries.iter().find_map(|e| match e {
            Entry::Tool(t) if &t.id == id => Some(t),
            _ => None,
        })
    }

    pub fn selected_change(&self) -> Option<&crate::workspace_changes::ChangedFile> {
        let choices = self.choices();
        let index = choices.get(self.selected)?.value.parse::<usize>().ok()?;
        self.workspace_changes.as_ref()?.files.get(index)
    }

    pub fn selected_attachment(&self) -> Option<&crate::attachments::Attachment> {
        if self.overlay != Some(Overlay::Attachments) {
            return None;
        }
        let choices = self.choices();
        let index = choices.get(self.selected)?.value.parse::<usize>().ok()?;
        self.attachments.get(index)
    }
}

fn reasoning_choices(metadata: &Value) -> Vec<Choice> {
    let Some(levels) = metadata.get("reasoningLevels") else {
        return Vec::new();
    };
    let selection = metadata["reasoningSelection"].as_str();
    let default = metadata["reasoningDefault"]
        .as_str()
        .filter(|level| !level.trim().is_empty());
    let detail = match levels.as_array() {
        None => "Selectable levels are unknown for this route",
        Some(levels) if levels.is_empty() => {
            "This route does not support reasoning-level selection"
        }
        Some(_) => "Use the model's request defaults",
    };
    let mut choices = vec![Choice {
        label: default.map_or_else(
            || "Model default".into(),
            |level| format!("Model default ({})", clean(level)),
        ),
        detail: if selection.is_none() {
            format!("Current · {detail}")
        } else {
            detail.into()
        },
        value: String::new(),
    }];
    choices.extend(levels.as_array().into_iter().flatten().filter_map(|level| {
        let level = level
            .as_str()
            .filter(|level| !level.trim().is_empty() && level.len() <= 256)?;
        Some(Choice {
            label: clean(level),
            detail: if selection == Some(level) {
                "Current · applies to subsequent requests"
            } else {
                "Apply to subsequent requests in this session"
            }
            .into(),
            value: level.into(),
        })
    }));
    choices
}

fn skill_name_character(character: char) -> bool {
    character.is_alphanumeric() || matches!(character, '_' | '.' | ':' | '-' | '[' | ']')
}

fn markdown_container_content(mut line: &str) -> (&str, usize) {
    let mut quotes = 0;
    loop {
        line = line.trim_start();
        if let Some(content) = line.strip_prefix('>') {
            quotes += 1;
            line = content;
            continue;
        }
        if let Some(content) = ["- ", "* ", "+ ", "-\t", "*\t", "+\t"]
            .iter()
            .find_map(|prefix| line.strip_prefix(prefix))
        {
            line = content;
            continue;
        }
        let digits = line.bytes().take_while(u8::is_ascii_digit).count();
        if digits > 0
            && [". ", ") ", ".\t", ")\t"]
                .iter()
                .any(|suffix| line[digits..].starts_with(suffix))
        {
            line = &line[digits + 2..];
            continue;
        }
        return (line, quotes);
    }
}

fn skill_reference_in_code(lines: &[String], row: usize, column: usize) -> bool {
    if lines[row].starts_with("    ") || lines[row].trim_start_matches(' ').starts_with('\t') {
        return true;
    }
    let mut fence: Option<(char, usize, usize)> = None;
    let mut inline_ticks = 0;
    for (index, line) in lines.iter().enumerate().take(row + 1) {
        let (trimmed, quotes) = if let Some((_, _, quotes)) = fence {
            // Only the opening container can prefix a closing fence.
            let mut content = Some(line.as_str());
            for _ in 0..quotes {
                content = content.and_then(|value| value.trim_start().strip_prefix('>'));
            }
            (content.unwrap_or("").trim_start(), quotes)
        } else {
            markdown_container_content(line)
        };
        let marker = trimmed.chars().next().unwrap_or(' ');
        let length = trimmed.chars().take_while(|value| *value == marker).count();
        if matches!(marker, '`' | '~') && length >= 3 {
            if let Some((open, minimum, _)) = fence {
                if marker == open && length >= minimum && trimmed[length..].trim().is_empty() {
                    fence = None;
                }
            } else if inline_ticks == 0 {
                fence = Some((marker, length, quotes));
            }
            if index == row {
                return true;
            }
            continue;
        }
        if fence.is_some() {
            if index == row {
                return true;
            }
            continue;
        }
        let characters: Vec<_> = line.chars().collect();
        let limit = if index == row {
            column
        } else {
            characters.len()
        };
        let mut cursor = 0;
        while cursor < limit {
            if characters[cursor] == '\\' && inline_ticks == 0 {
                cursor += 2;
                continue;
            }
            if characters[cursor] != '`' {
                cursor += 1;
                continue;
            }
            let run = characters[cursor..limit]
                .iter()
                .take_while(|value| **value == '`')
                .count();
            if inline_ticks == 0 {
                inline_ticks = run;
            } else if inline_ticks == run {
                inline_ticks = 0;
            }
            cursor += run;
        }
    }
    fence.is_some() || inline_ticks != 0
}

pub fn default_commands() -> Vec<Command> {
    [
        ("/new", "Save and start a new conversation"),
        ("/sessions", "Find and resume a saved conversation"),
        ("/drafts", "Choose saved input to restore"),
        ("/resume", "Resume a session by ID or prefix"),
        ("/rewind", "Return to an earlier message and edit it"),
        ("/tree", "Explore and continue any conversation branch"),
        ("/tools", "Browse tool calls and their full output"),
        (
            "/changes",
            "Inspect Git workspace changes and conversation edits",
        ),
        ("/queue", "Edit or remove messages waiting to be sent"),
        ("/steer", "Send a correction to the running task (Alt+S)"),
        ("/attach", "Search workspace files or attach a file path"),
        (
            "/attachments",
            "Review or remove the current draft's attachments",
        ),
        (
            "/reconnect",
            "Reconnect the backend and restore this conversation",
        ),
        (
            "/retry",
            "Restore interrupted input for editing before resending",
        ),
        ("/undo", "Review and undo the last file rollback"),
        ("/model", "Choose model and reasoning effort (Alt+P)"),
        (
            "/effort",
            "Adjust reasoning effort; auto restores model defaults",
        ),
        (
            "/reasoning",
            "Choose reasoning effort; default restores model defaults",
        ),
        ("/mode", "Choose Code or Plan mode"),
        ("/plan", "Enter planning mode with execution disabled"),
        ("/code", "Return to Code mode"),
        ("/permissions", "Choose Auto, Manual, Accept edits or Plan"),
        (
            "/sandbox",
            "Choose Python sandbox; on or off changes this session",
        ),
        ("/tasks", "Inspect plan steps and background jobs (Alt+T)"),
        (
            "/skills",
            "Browse available skills, active status and documentation",
        ),
        (
            "/mcp",
            "Inspect MCP servers; reconnect <name> retries discovery",
        ),
        ("/stop", "Stop a selected background job by its task ID"),
        (
            "/search",
            "Find a message or tool output in this conversation",
        ),
        ("/status", "Show workspace, model and session"),
        (
            "/hud",
            "Configure status rows, layout, colors and usage sources",
        ),
        (
            "/context",
            "Read system prompts, messages and model output; all lists blocks",
        ),
        ("/logs", "Inspect backend diagnostics"),
        ("/help", "Keyboard shortcuts and commands"),
        ("/cancel", "Stop the current request"),
        ("/exit", "Save and exit"),
    ]
    .into_iter()
    .map(|(name, description)| Command {
        name: name.into(),
        description: description.into(),
        ..Command::default()
    })
    .collect()
}

/// Keep untrusted output from injecting terminal control sequences.
pub fn clean(text: &str) -> String {
    TerminalTextFilter::default().feed(text)
}

#[derive(Default)]
pub(crate) struct TerminalTextFilter {
    state: u8,
    osc: bool,
}

impl TerminalTextFilter {
    pub(crate) fn feed(&mut self, text: &str) -> String {
        let mut visible = String::new();
        for character in text.chars() {
            if matches!(self.state, 3 | 4) {
                if matches!(character, '\u{18}' | '\u{1a}' | '\u{9c}')
                    || (self.osc && character == '\u{7}')
                    || (self.state == 4 && character == '\\')
                {
                    self.state = 0;
                } else {
                    self.state = if character == '\u{1b}' { 4 } else { 3 };
                }
                continue;
            }
            match character {
                '\u{1b}' => {
                    self.state = 1;
                    continue;
                }
                '\u{9b}' => {
                    self.state = 2;
                    continue;
                }
                '\u{90}' | '\u{98}' | '\u{9d}' | '\u{9e}' | '\u{9f}' => {
                    self.state = 3;
                    self.osc = character == '\u{9d}';
                    continue;
                }
                '\u{18}' | '\u{1a}' | '\u{9c}' => {
                    self.state = 0;
                    continue;
                }
                _ => {}
            }
            if self.state == 5 {
                if ('0'..='~').contains(&character) {
                    self.state = 0;
                }
                continue;
            }
            if self.state == 1 {
                if character == '[' {
                    self.state = 2;
                    continue;
                }
                if matches!(character, 'P' | 'X' | ']' | '^' | '_') {
                    self.state = 3;
                    self.osc = character == ']';
                    continue;
                }
                if (' '..='/').contains(&character) {
                    self.state = 5;
                    continue;
                }
                if ('0'..='~').contains(&character) {
                    self.state = 0;
                    continue;
                }
            } else if self.state == 2 {
                if ('@'..='~').contains(&character) {
                    self.state = 0;
                    continue;
                }
                if (' '..='?').contains(&character) {
                    continue;
                }
            }
            if character == '\n'
                || character == '\t'
                || (!character.is_control()
                    && !matches!(character, '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}'))
            {
                self.state = 0;
                visible.push(character);
            }
        }
        visible
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strips_control_payloads_and_handles_split_sequences() {
        let examples = [
            ("你好\tthere\nnext\r\0\x07\x08\x7f", "你好\tthere\nnext"),
            ("\x1b[2J\x1b[1;1H\x1b[31mred\x1b[0m", "red"),
            ("\x1b]52;c;c2VjcmV0\x07visible", "visible"),
            (
                "\x1b]8;;https://example.com\x1b\\label\x1b]8;;\x1b\\",
                "label",
            ),
            ("a\x1bPprivate payload\x1b\\b\x1b^hidden\x1b\\c", "abc"),
            ("\u{9d}52;c;hidden\u{9c}\u{9b}31mred\u{9b}0m", "red"),
            ("\x1b(Bnormal\x1b(P text\x1b7 saved", "normal text saved"),
            ("visible\x1b]52;c;unterminated", "visible"),
        ];
        for (source, expected) in examples {
            assert_eq!(clean(source), expected);
            let mut filter = TerminalTextFilter::default();
            let actual: String = source
                .chars()
                .map(|c| filter.feed(&c.to_string()))
                .collect();
            assert_eq!(actual, expected);
        }
    }
}
