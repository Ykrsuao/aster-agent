// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Read-only selection uses source bytes, while rendering and hit testing use terminal cells.

use std::{ops::Range, sync::Arc};

use crossterm::event::{KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::{
    buffer::Buffer,
    layout::{Position, Rect},
    style::Style,
    text::{Line, Span},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use crate::state::Overlay;

pub const SCREEN: u16 = 0;
pub const TRANSCRIPT: u16 = 1;

#[derive(Clone, Debug)]
pub struct RowSource {
    pub text: Arc<str>,
    pub range: Range<usize>,
    pub columns: usize,
}

impl RowSource {
    pub fn whole(text: String) -> Self {
        let range = 0..text.len();
        Self {
            text: text.into(),
            range,
            columns: usize::MAX,
        }
    }
}

pub struct Fragment {
    pub line: Line<'static>,
    pub source: RowSource,
}

pub(crate) fn glyph_width(glyph: &str, columns: usize) -> usize {
    if glyph == "\t" {
        4.min(columns)
    } else {
        let width = glyph.width();
        if width > columns {
            1
        } else {
            width
        }
    }
}

pub(crate) fn display_glyph(glyph: &str, columns: usize) -> String {
    if glyph == "\t" {
        " ".repeat(4.min(columns))
    } else if glyph.width() > columns {
        "�".into()
    } else {
        glyph.into()
    }
}

/// Keep graphemes and styles together, retaining the source of every soft-wrapped row.
pub fn wrap(line: Line<'static>, columns: usize) -> Vec<Fragment> {
    if columns == 0 {
        return Vec::new();
    }
    let base = line.style;
    let mut text = String::new();
    let mut styles = Vec::new();
    for span in line.spans {
        let from = text.len();
        text.push_str(&span.content);
        styles.push((from..text.len(), base.patch(span.style)));
    }
    let text: Arc<str> = text.into();
    let mut rows = Vec::new();
    let mut offset = 0;
    let mut style_index = 0;
    for hard_line in text.split('\n') {
        let glyphs: Vec<_> = hard_line
            .grapheme_indices(true)
            .map(|(byte, glyph)| {
                let from = offset + byte;
                while style_index + 1 < styles.len() && styles[style_index].0.end <= from {
                    style_index += 1;
                }
                (
                    from,
                    glyph,
                    styles.get(style_index).map_or(base, |(_, style)| *style),
                )
            })
            .collect();
        if glyphs.is_empty() {
            rows.push(Fragment {
                line: Line::styled("", base),
                source: RowSource {
                    text: text.clone(),
                    range: offset..offset,
                    columns,
                },
            });
        }
        let mut start = 0;
        while start < glyphs.len() {
            let mut end = start;
            let mut used = 0;
            let mut break_at = None;
            while let Some((_, glyph, _)) = glyphs.get(end) {
                let width = glyph_width(glyph, columns);
                if used + width > columns {
                    break;
                }
                used += width;
                if glyph.chars().all(char::is_whitespace) {
                    break_at = Some(end + 1);
                }
                end += 1;
            }
            if end < glyphs.len() {
                if let Some(boundary) = break_at.filter(|boundary| *boundary > start + 1) {
                    end = boundary;
                }
            }
            let mut spans: Vec<Span<'static>> = Vec::new();
            for (_, glyph, style) in &glyphs[start..end] {
                let glyph = display_glyph(glyph, columns);
                if let Some(last) = spans.last_mut().filter(|span| span.style == *style) {
                    last.content.to_mut().push_str(&glyph);
                } else {
                    spans.push(Span::styled(glyph, *style));
                }
            }
            rows.push(Fragment {
                line: Line::from(spans).style(base),
                source: RowSource {
                    text: text.clone(),
                    range: glyphs[start].0
                        ..glyphs
                            .get(end)
                            .map_or(offset + hard_line.len(), |glyph| glyph.0),
                    columns,
                },
            });
            start = end;
        }
        offset += hard_line.len() + 1;
    }
    rows
}

#[derive(Debug)]
struct Glyph {
    source: Range<usize>,
    columns: Range<usize>,
}

#[derive(Debug)]
struct Row {
    source: Range<usize>,
    gutter: usize,
    glyphs: Vec<Glyph>,
}

#[derive(Debug)]
pub struct Document {
    text: String,
    rows: Vec<Row>,
}

impl Document {
    pub fn new(rows: impl IntoIterator<Item = (RowSource, usize)>, columns: usize) -> Self {
        let mut document = Self {
            text: String::new(),
            rows: Vec::new(),
        };
        let mut previous: Option<RowSource> = None;
        for (source, gutter) in rows {
            let continued = previous.as_ref().is_some_and(|previous| {
                Arc::ptr_eq(&previous.text, &source.text)
                    && previous.range.end == source.range.start
                    && source.range.start > 0
            });
            if !document.rows.is_empty() && !continued {
                document.text.push('\n');
            }
            let from = document.text.len();
            let text = &source.text[source.range.clone()];
            document.text.push_str(text);
            let mut column = gutter;
            let available = columns.saturating_sub(gutter).min(source.columns).max(1);
            let glyphs = text
                .grapheme_indices(true)
                .map(|(byte, glyph)| {
                    let start = column;
                    column += glyph_width(glyph, available);
                    Glyph {
                        source: from + byte..from + byte + glyph.len(),
                        columns: start..column,
                    }
                })
                .collect();
            document.rows.push(Row {
                source: from..document.text.len(),
                gutter,
                glyphs,
            });
            previous = Some(source);
        }
        document
    }

    /// Static labels also remain selectable; borders and native-image payloads are excluded.
    pub fn from_buffer(buffer: &Buffer, area: Rect) -> Self {
        let mut rows = Vec::new();
        for y in area.y..area.bottom() {
            let mut text = String::new();
            let mut continuation = 0;
            for x in area.x..area.right() {
                if continuation > 0 {
                    continuation -= 1;
                    continue;
                }
                let cell = &buffer[(x, y)];
                let symbol = cell.symbol();
                let width = symbol.width().max(1);
                if cell.diff_option == ratatui::buffer::CellDiffOption::Skip
                    || symbol.contains('\x1b')
                    || symbol
                        .chars()
                        .any(|c| ('\u{2500}'..='\u{257f}').contains(&c))
                {
                    text.push_str(&" ".repeat(width));
                } else {
                    text.push_str(symbol);
                }
                continuation = width.saturating_sub(1);
            }
            let gutter = text[..text.len() - text.trim_start().len()].width();
            rows.push((RowSource::whole(text.trim().to_owned()), gutter));
        }
        Self::new(rows, usize::from(area.width))
    }

    fn offset_at(&self, row: usize, column: usize) -> usize {
        let Some(row) = self.rows.get(row).or_else(|| self.rows.last()) else {
            return 0;
        };
        for glyph in &row.glyphs {
            if glyph.columns.end > column {
                return glyph.source.start;
            }
        }
        row.source.end
    }

    fn hit(&self, row: usize, column: usize) -> bool {
        self.rows.get(row).is_some_and(|row| {
            row.glyphs
                .iter()
                .any(|glyph| glyph.columns.contains(&column))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScrollTarget {
    Transcript,
    Detail,
}

pub struct Region {
    pub id: u16,
    pub area: Rect,
    pub top: usize,
    pub max_top: usize,
    pub scroll: Option<ScrollTarget>,
    pub document: Arc<Document>,
    pub excluded: Vec<Rect>,
}

impl Region {
    fn hit(&self, position: Position) -> bool {
        self.area.contains(position)
            && !self.excluded.iter().any(|area| area.contains(position))
            && (self.id != SCREEN
                || self.document.hit(
                    self.top + usize::from(position.y - self.area.y),
                    usize::from(position.x - self.area.x),
                ))
    }

    fn offset(&self, position: Position, top: usize) -> usize {
        let row = top
            + usize::from(position.y.saturating_sub(self.area.y))
                .min(usize::from(self.area.height).saturating_sub(1));
        let column =
            usize::from(position.x.saturating_sub(self.area.x)).min(usize::from(self.area.width));
        self.document.offset_at(row, column)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub conversation: u64,
    pub overlay: Option<Overlay>,
    pub subject: String,
    pub query: String,
}

#[derive(Debug)]
struct Press {
    event: MouseEvent,
    area: Rect,
    moved: bool,
}

#[derive(Debug)]
struct Selection {
    scope: Scope,
    region: u16,
    document: Arc<Document>,
    anchor: usize,
    focus: usize,
    viewport: Rect,
    press: Option<Press>,
}

impl Selection {
    fn range(&self) -> Range<usize> {
        self.anchor.min(self.focus)..self.anchor.max(self.focus)
    }
}

#[derive(Default)]
pub struct SelectionState {
    selection: Option<Selection>,
}

#[derive(Default)]
pub struct Outcome {
    pub handled: bool,
    pub changed: bool,
    pub copy: Option<String>,
    pub click: Option<MouseEvent>,
    pub scroll: Option<(ScrollTarget, usize, usize)>,
}

impl SelectionState {
    pub fn clear(&mut self) {
        self.selection = None;
    }

    pub fn cancel_pointer(&mut self) {
        if let Some(selection) = self.selection.as_mut() {
            selection.press = None;
        }
    }

    pub fn selected_text(&self, scope: &Scope) -> Option<String> {
        let selection = self
            .selection
            .as_ref()
            .filter(|selection| selection.scope == *scope)?;
        let range = selection.range();
        if range.is_empty() {
            return None;
        }
        selection.document.text.get(range).map(str::to_owned)
    }

    pub fn pinning_transcript(&self, scope: &Scope) -> bool {
        self.selection.as_ref().is_some_and(|selection| {
            selection.scope == *scope
                && selection.region == TRANSCRIPT
                && (selection.press.is_some() || selection.anchor != selection.focus)
        })
    }

    pub fn transcript_viewport(&self, scope: &Scope) -> Option<Rect> {
        self.selection
            .as_ref()
            .filter(|selection| {
                selection.scope == *scope
                    && selection.region == TRANSCRIPT
                    && (selection.press.is_some() || selection.anchor != selection.focus)
            })
            .map(|selection| selection.viewport)
    }

    /// Keep source positions across reflow and appended output; reject changed views or prefixes.
    pub fn sync(&mut self, scope: &Scope, regions: &[Region]) {
        let Some(selection) = self.selection.as_mut() else {
            return;
        };
        let region = regions.iter().find(|region| region.id == selection.region);
        if selection.scope != *scope
            || region.is_none_or(|region| {
                let end = selection.range().end;
                region.document.text.get(..end) != selection.document.text.get(..end)
            })
        {
            self.clear();
            return;
        }
        let region = region.expect("selection region exists");
        if selection
            .press
            .as_ref()
            .is_some_and(|press| press.area != region.area)
        {
            selection.press = None;
        }
        selection.document = region.document.clone();
        selection.viewport = region.area;
    }

    pub fn mouse(&mut self, scope: &Scope, regions: &[Region], event: MouseEvent) -> Outcome {
        self.sync(scope, regions);
        let position = Position::new(event.column, event.row);
        if event.kind == MouseEventKind::Down(MouseButton::Left) {
            let Some(region) = regions.iter().rev().find(|region| region.hit(position)) else {
                let changed = self.selection.take().is_some();
                return Outcome {
                    changed,
                    ..Outcome::default()
                };
            };
            let focus = region.offset(position, region.top);
            let anchor = self
                .selection
                .as_ref()
                .filter(|selection| {
                    event.modifiers.contains(KeyModifiers::SHIFT) && selection.region == region.id
                })
                .map_or(focus, |selection| selection.anchor);
            self.selection = Some(Selection {
                scope: scope.clone(),
                region: region.id,
                document: region.document.clone(),
                anchor,
                focus,
                viewport: region.area,
                press: Some(Press {
                    event,
                    area: region.area,
                    moved: false,
                }),
            });
            return Outcome {
                handled: true,
                changed: true,
                ..Outcome::default()
            };
        }
        if matches!(event.kind, MouseEventKind::Down(_)) {
            self.cancel_pointer();
            return Outcome::default();
        }
        let Some(selection) = self
            .selection
            .as_mut()
            .filter(|selection| selection.press.is_some())
        else {
            return Outcome::default();
        };
        let Some(region) = regions.iter().find(|region| region.id == selection.region) else {
            return Outcome::default();
        };
        let mut top = region.top;
        let mut outcome = Outcome {
            handled: true,
            changed: true,
            ..Outcome::default()
        };
        match event.kind {
            MouseEventKind::Drag(MouseButton::Left) => {
                selection.press.as_mut().expect("captured pointer").moved = true;
                if position.y < region.area.y {
                    top = top.saturating_sub(usize::from(region.area.y - position.y));
                } else if position.y >= region.area.bottom() {
                    top = top
                        .saturating_add(usize::from(position.y - region.area.bottom()) + 1)
                        .min(region.max_top);
                }
            }
            MouseEventKind::ScrollUp => {
                selection.press.as_mut().expect("captured pointer").moved = true;
                top = top.saturating_sub(3);
            }
            MouseEventKind::ScrollDown => {
                selection.press.as_mut().expect("captured pointer").moved = true;
                top = top.saturating_add(3).min(region.max_top);
            }
            MouseEventKind::Up(MouseButton::Left) => {
                let press = selection.press.take().expect("captured pointer");
                if region.area.contains(position) {
                    selection.focus = region.offset(position, top);
                }
                let range = selection.range();
                if !range.is_empty() {
                    outcome.copy = selection.document.text.get(range).map(str::to_owned);
                } else if !press.moved && region.hit(position) {
                    outcome.click = Some(press.event);
                }
                return outcome;
            }
            _ => return Outcome::default(),
        }
        if let Some(target) = region.scroll {
            if top != region.top {
                outcome.scroll = Some((target, top, region.max_top));
            }
        } else {
            top = region.top;
        }
        selection.focus = region.offset(position, top);
        outcome
    }

    pub fn paint(&self, regions: &[Region], buffer: &mut Buffer, style: Style) {
        let Some(selection) = self.selection.as_ref() else {
            return;
        };
        let Some(region) = regions.iter().find(|region| region.id == selection.region) else {
            return;
        };
        let selected = selection.range();
        if selected.is_empty() {
            return;
        }
        for (index, row) in region
            .document
            .rows
            .iter()
            .skip(region.top)
            .take(usize::from(region.area.height))
            .enumerate()
        {
            let y = region.area.y + index as u16;
            for glyph in &row.glyphs {
                if glyph.source.start >= selected.end || glyph.source.end <= selected.start {
                    continue;
                }
                for column in glyph.columns.clone() {
                    if column >= usize::from(region.area.width) {
                        break;
                    }
                    let position = Position::new(region.area.x + column as u16, y);
                    if !region.excluded.iter().any(|area| area.contains(position)) {
                        buffer[position].set_style(style);
                    }
                }
            }
            if selected.start <= row.source.end
                && selected.end > row.source.end
                && region.document.text.as_bytes().get(row.source.end) == Some(&b'\n')
            {
                let x = row
                    .glyphs
                    .last()
                    .map_or(row.gutter, |glyph| glyph.columns.end);
                if x < usize::from(region.area.width) {
                    let position = Position::new(region.area.x + x as u16, y);
                    if !region.excluded.iter().any(|area| area.contains(position)) {
                        buffer[position].set_style(style);
                    }
                }
            }
        }
    }
}

#[derive(Default)]
pub struct Cache {
    entries: Vec<(u16, u64, Arc<Document>)>,
}

impl Cache {
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn get(
        &mut self,
        id: u16,
        signature: u64,
        prepare: impl FnOnce() -> Document,
    ) -> Arc<Document> {
        if let Some((_, previous, document)) =
            self.entries.iter_mut().find(|(key, _, _)| *key == id)
        {
            if *previous != signature {
                *document = Arc::new(prepare());
                *previous = signature;
            }
            return document.clone();
        }
        let document = Arc::new(prepare());
        self.entries.push((id, signature, document.clone()));
        document
    }
}
