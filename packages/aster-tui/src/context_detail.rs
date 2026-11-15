// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Read-only text tied to the exact context request selected by the user.

use std::ops::Range;

use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::{
    layout::{Position, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
};
use serde_json::{json, Value};
use unicode_segmentation::UnicodeSegmentation;

use crate::context::Category;
use crate::state::clean;
use crate::text_selection::{display_glyph, glyph_width, wrap};

const MAX_CONTENT_CHARS: usize = 1_000_000;

#[derive(Clone, Debug)]
pub struct Item {
    pub category: Category,
    pub key: String,
    pub characters: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Category(Category),
    Block { category: Category, key: String },
    Output,
}

impl Target {
    pub fn label(&self) -> String {
        match self {
            Self::Category(category) => category.label().into(),
            Self::Block { key, .. } => clean(key),
            Self::Output => "Latest model output".into(),
        }
    }

    pub fn params(&self, request_id: &str) -> Value {
        let (category, key) = match self {
            Self::Category(category) => (category.id(), None),
            Self::Block { category, key } => (category.id(), Some(key.as_str())),
            Self::Output => ("output", None),
        };
        json!({"requestId": request_id, "category": category, "key": key})
    }
}

#[derive(Debug)]
struct Section {
    key: String,
    text: String,
    characters: u64,
    captured: usize,
    truncated: bool,
}

#[derive(Debug)]
struct Wrapped {
    columns: usize,
    base: Style,
    muted: Style,
    rows: Vec<Line<'static>>,
    /// Source byte ranges exclude soft wraps and retain real newlines and tabs.
    ranges: Vec<Range<usize>>,
    text: String,
}

impl Wrapped {
    fn append(&mut self, text: &str, style: Style) {
        if !self.rows.is_empty() {
            self.text.push('\n');
        }
        let offset = self.text.len();
        self.text.push_str(text);
        // Line::styled strips newlines; keep them in a span for the shared wrapper.
        let line = Line::from(Span::raw(text.to_owned())).style(style);
        for fragment in wrap(line, self.columns) {
            self.rows.push(fragment.line);
            self.ranges
                .push(offset + fragment.source.range.start..offset + fragment.source.range.end);
        }
    }

    fn offset_at(&self, row: usize, column: usize) -> usize {
        let Some(range) = self.ranges.get(row).or_else(|| self.ranges.last()) else {
            return 0;
        };
        let mut used = 0;
        for (byte, glyph) in self.text[range.clone()].grapheme_indices(true) {
            let cells = glyph_width(glyph, self.columns);
            if used + cells > column {
                return range.start + byte;
            }
            used += cells;
        }
        range.end
    }
}

#[derive(Debug)]
struct Selection {
    anchor: usize,
    focus: usize,
}

impl Selection {
    fn range(&self) -> Range<usize> {
        self.anchor.min(self.focus)..self.anchor.max(self.focus)
    }
}

#[derive(Debug)]
pub struct View {
    pub generation: u64,
    pub request_id: String,
    pub target: Target,
    pub return_scroll: u16,
    pub scroll: usize,
    loading: bool,
    error: Option<String>,
    sections: Vec<Section>,
    wrapped: Option<Wrapped>,
    selection: Option<Selection>,
    dragging: bool,
}

impl View {
    pub fn new(request_id: String, target: Target, generation: u64, return_scroll: u16) -> Self {
        Self {
            generation,
            request_id,
            target,
            return_scroll,
            loading: true,
            error: None,
            sections: Vec::new(),
            wrapped: None,
            selection: None,
            dragging: false,
            scroll: 0,
        }
    }

    pub fn complete(&mut self, value: &Value) {
        self.loading = false;
        self.wrapped = None;
        self.selection = None;
        self.cancel_pointer();
        self.error = None;
        self.sections.clear();
        if let Some(error) = value.get("error") {
            self.error = Some(clean(
                error["data"]["reason"]
                    .as_str()
                    .or_else(|| error["message"].as_str())
                    .unwrap_or("Could not load this content."),
            ));
            return;
        }
        let result = &value["result"];
        let expected = self.target.params(&self.request_id);
        let source = if self.target == Target::Output {
            "model_response"
        } else {
            "model_request"
        };
        if result["requestId"] != expected["requestId"]
            || result["category"] != expected["category"]
            || result["key"] != expected["key"]
            || result["source"] != source
        {
            self.error =
                Some("This content belongs to a different request. Go back and reopen it.".into());
            return;
        }
        let Some(sections) = result["sections"].as_array().filter(|sections| {
            sections.len() <= 64 && (!sections.is_empty() || self.target == Target::Output)
        }) else {
            self.error = Some("Invalid context content response.".into());
            return;
        };
        let mut captured = 0;
        for section in sections {
            let (Some(key), Some(text), Some(characters), Some(truncated)) = (
                section["key"].as_str(),
                section["text"].as_str(),
                section["characters"].as_u64(),
                section["truncated"].as_bool(),
            ) else {
                self.error = Some("Invalid context content response.".into());
                break;
            };
            let count = text.chars().count();
            captured += count;
            if key.chars().count() > 120
                || captured > MAX_CONTENT_CHARS
                || characters < count as u64
                || (!truncated && characters != count as u64)
            {
                self.error = Some("Invalid or oversized context content response.".into());
                break;
            }
            self.sections.push(Section {
                key: clean(key),
                text: clean(text),
                characters,
                captured: count,
                truncated,
            });
        }
        if self.error.is_some() {
            self.sections.clear();
        }
    }

    pub fn rows(&mut self, columns: usize, base: Style, muted: Style) -> &[Line<'static>] {
        let columns = columns.max(1);
        if self.wrapped.as_ref().is_none_or(|cached| {
            cached.columns != columns || cached.base != base || cached.muted != muted
        }) {
            let mut wrapped = Wrapped {
                columns,
                base,
                muted,
                rows: Vec::new(),
                ranges: Vec::new(),
                text: String::new(),
            };
            if self.loading {
                wrapped.append("Loading content…", muted);
            } else if let Some(error) = &self.error {
                wrapped.append(error, base);
            } else {
                wrapped.append(
                    if self.target == Target::Output {
                        "Actual model output for the selected request."
                    } else {
                        "Actual text sent to the model in the selected request."
                    },
                    muted,
                );
                for section in &self.sections {
                    wrapped.append("", base);
                    wrapped.append(&section.key, base.add_modifier(Modifier::BOLD));
                    if section.truncated {
                        wrapped.append(
                            &format!(
                                "Showing {} of {} characters (capture limit).",
                                section.captured, section.characters
                            ),
                            muted,
                        );
                    }
                    wrapped.append(&section.text, base);
                }
                if self.sections.is_empty() {
                    wrapped.append("The model returned no text or tool calls.", muted);
                }
            }
            self.wrapped = Some(wrapped);
        }
        &self
            .wrapped
            .as_ref()
            .expect("content rows were prepared")
            .rows
    }

    pub fn has_selection(&self) -> bool {
        self.selection.as_ref().is_some_and(|s| s.anchor != s.focus)
    }

    pub fn selected_text(&self) -> Option<String> {
        let range = self.selection.as_ref()?.range();
        if range.is_empty() {
            return None;
        }
        self.wrapped.as_ref()?.text.get(range).map(str::to_owned)
    }

    /// Apply selection only to visible rows; wrapping stays cached during drags.
    pub fn painted_row(&self, index: usize, selected: Style) -> Option<Line<'static>> {
        let wrapped = self.wrapped.as_ref()?;
        let line = wrapped.rows.get(index)?;
        let Some(selection) = self.selection.as_ref().map(Selection::range) else {
            return Some(line.clone());
        };
        let range = &wrapped.ranges[index];
        let newline_selected = selection.start <= range.end
            && selection.end > range.end
            && wrapped.text.as_bytes().get(range.end) == Some(&b'\n');
        if selection.end <= range.start || (selection.start >= range.end && !newline_selected) {
            return Some(line.clone());
        }
        let mut spans: Vec<Span<'static>> = Vec::new();
        let mut used = 0;
        for (byte, glyph) in wrapped.text[range.clone()].grapheme_indices(true) {
            let from = range.start + byte;
            let style = if from < selection.end && from + glyph.len() > selection.start {
                line.style.patch(selected)
            } else {
                line.style
            };
            let display = display_glyph(glyph, wrapped.columns);
            if let Some(last) = spans.last_mut().filter(|span| span.style == style) {
                last.content.to_mut().push_str(&display);
            } else {
                spans.push(Span::styled(display, style));
            }
            used += glyph_width(glyph, wrapped.columns);
        }
        if newline_selected && used < wrapped.columns {
            spans.push(Span::styled(" ", line.style.patch(selected)));
        }
        Some(Line::from(spans).style(line.style))
    }

    pub fn cancel_pointer(&mut self) {
        self.dragging = false;
    }

    pub fn clear_selection(&mut self) {
        self.selection = None;
        self.cancel_pointer();
    }

    /// Hit-test against the same source ranges as the renderer, including wide glyphs.
    pub fn mouse(&mut self, area: Rect, mouse: MouseEvent) -> bool {
        if area.is_empty()
            || self.loading
            || self.error.is_some()
            || self
                .wrapped
                .as_ref()
                .is_none_or(|w| w.columns != usize::from(area.width))
        {
            self.cancel_pointer();
            return false;
        }
        let inside = area.contains(Position::new(mouse.column, mouse.row));
        let wrapped = self.wrapped.as_ref().expect("content was rendered");
        let max_top = wrapped.rows.len().saturating_sub(usize::from(area.height));
        self.scroll = self.scroll.min(max_top);
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) if inside => {}
            MouseEventKind::Drag(MouseButton::Left) if self.dragging => {
                if mouse.row < area.y {
                    self.scroll = self.scroll.saturating_sub(usize::from(area.y - mouse.row));
                } else if mouse.row >= area.bottom() {
                    self.scroll = self
                        .scroll
                        .saturating_add(usize::from(mouse.row - area.bottom()) + 1)
                        .min(max_top);
                }
            }
            MouseEventKind::Up(MouseButton::Left) if self.dragging => {
                self.dragging = false;
                if !inside {
                    return true;
                }
            }
            MouseEventKind::ScrollUp | MouseEventKind::ScrollDown if inside || self.dragging => {
                self.scroll = if mouse.kind == MouseEventKind::ScrollUp {
                    self.scroll.saturating_sub(3)
                } else {
                    self.scroll.saturating_add(3).min(max_top)
                };
                if !self.dragging {
                    return true;
                }
            }
            MouseEventKind::Down(_) => {
                self.cancel_pointer();
                if !inside {
                    return self.selection.take().is_some();
                }
                return false;
            }
            _ => return false,
        }
        let row = self.scroll
            + usize::from(mouse.row.saturating_sub(area.y))
                .min(usize::from(area.height).saturating_sub(1));
        let column = usize::from(mouse.column.saturating_sub(area.x)).min(wrapped.columns);
        let focus = wrapped.offset_at(row, column);
        if mouse.kind == MouseEventKind::Down(MouseButton::Left) {
            let anchor = if mouse.modifiers.contains(KeyModifiers::SHIFT) {
                self.selection
                    .as_ref()
                    .map_or(focus, |selection| selection.anchor)
            } else {
                focus
            };
            self.selection = Some(Selection { anchor, focus });
            self.dragging = true;
        } else if let Some(selection) = self.selection.as_mut() {
            selection.focus = focus;
        }
        true
    }
}
