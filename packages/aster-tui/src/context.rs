// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Provider totals and approximate request composition for the local context inspector.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use serde_json::Value;
use unicode_width::UnicodeWidthStr;

use crate::attachments::Attachment;
use crate::context_detail::{Item, Target};
use crate::state::clean;
use crate::terminal_image::CellPixels;

const CELLS: usize = 100;
const CELL: &str = "■";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    SystemPrompt,
    SystemTools,
    MemoryFiles,
    Skills,
    Messages,
    Runtime,
    Other,
}

impl Category {
    pub(crate) fn parse(id: &str) -> Option<Self> {
        Some(match id {
            "system_prompt" => Self::SystemPrompt,
            "system_tools" => Self::SystemTools,
            "memory_files" => Self::MemoryFiles,
            "skills" => Self::Skills,
            "messages" => Self::Messages,
            "runtime_context" => Self::Runtime,
            "other" => Self::Other,
            _ => return None,
        })
    }

    pub(crate) fn id(self) -> &'static str {
        match self {
            Self::SystemPrompt => "system_prompt",
            Self::SystemTools => "system_tools",
            Self::MemoryFiles => "memory_files",
            Self::Skills => "skills",
            Self::Messages => "messages",
            Self::Runtime => "runtime_context",
            Self::Other => "other",
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::SystemPrompt => "System prompt",
            Self::SystemTools => "System tools",
            Self::MemoryFiles => "Memory files",
            Self::Skills => "Skills",
            Self::Messages => "Messages",
            Self::Runtime => "Runtime context",
            Self::Other => "Other context",
        }
    }

    fn style(self, base: Style, monochrome: bool) -> Style {
        if monochrome {
            return base;
        }
        base.fg(match self {
            Self::SystemPrompt => Color::Rgb(184, 157, 205),
            Self::SystemTools => Color::Rgb(129, 170, 215),
            Self::MemoryFiles => Color::Rgb(220, 186, 115),
            Self::Skills => Color::Rgb(122, 186, 186),
            Self::Messages => Color::Rgb(147, 186, 133),
            Self::Runtime => Color::Rgb(205, 165, 129),
            Self::Other => Color::Rgb(200, 200, 200),
        })
    }
}

#[derive(Clone, Debug)]
pub struct CategoryUsage {
    pub category: Category,
    pub tokens: u64,
}

#[derive(Clone, Debug)]
pub struct ItemUsage {
    pub category: Category,
    pub key: String,
    pub tokens: u64,
}

#[derive(Clone, Debug)]
pub struct ContextSnapshot {
    pub used: Option<u64>,
    pub window: Option<u64>,
    pub reserved: Option<u64>,
    pub planning_reserve: bool,
    pub model: String,
    pub categories: Vec<CategoryUsage>,
    pub items: Vec<ItemUsage>,
    pub media: bool,
    pub request_id: Option<String>,
    pub content_items: Vec<Item>,
    pub output_available: bool,
}

#[derive(Clone, Copy)]
pub struct Browser {
    pub expanded: bool,
    pub selected: usize,
}

pub struct Rows {
    pub lines: Vec<Line<'static>>,
    pub targets: Vec<(usize, usize)>,
}

fn content_items(metadata: &Value) -> Vec<Item> {
    let Some(items) = metadata["contentItems"]
        .as_array()
        .filter(|items| items.len() <= 64)
    else {
        return Vec::new();
    };
    let mut parsed: Vec<Item> = Vec::new();
    for item in items {
        let (Some(category), Some(key), Some(characters)) = (
            item["category"].as_str().and_then(Category::parse),
            item["key"].as_str(),
            item["chars"].as_u64(),
        ) else {
            return Vec::new();
        };
        if key.is_empty()
            || key.chars().count() > 120
            || characters == 0
            || parsed
                .iter()
                .any(|item| item.category == category && item.key == key)
        {
            return Vec::new();
        }
        parsed.push(Item {
            category,
            key: key.to_owned(),
            characters,
        });
    }
    parsed
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Used(Option<Category>),
    Reserved,
    Free,
}

impl Cell {
    fn style(self, base: Style, muted: Style, monochrome: bool) -> Style {
        if monochrome {
            return match self {
                Self::Used(_) => base.add_modifier(Modifier::BOLD),
                Self::Reserved => muted,
                Self::Free => muted.add_modifier(Modifier::DIM),
            };
        }
        match self {
            Self::Used(Some(category)) => category.style(base, monochrome),
            Self::Used(None) => base,
            Self::Reserved => muted.fg(Color::Rgb(156, 168, 184)),
            Self::Free => muted.fg(Color::Rgb(98, 98, 98)),
        }
    }
}

/// A complete pixel grid, sized to the terminal's reported cell geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridLayout {
    pub columns: u16,
    pub rows: u16,
    pixel_width: usize,
    pixel_height: usize,
    grid_columns: usize,
    side: usize,
    gap: usize,
}

impl GridLayout {
    fn fit(columns: u16, rows: u16, pixels: CellPixels) -> Option<Self> {
        CellPixels::new(pixels.width, pixels.height)?;
        let side = (usize::from(pixels.width) * 7).div_ceil(5).clamp(8, 32);
        let gap = (side / 7).max(1);
        [20, 10, 5, 2, 1].into_iter().find_map(|grid_columns| {
            let pixel_width = grid_columns * (side + gap) - gap;
            let pixel_height = CELLS / grid_columns * (side + gap) - gap;
            let width = pixel_width.div_ceil(usize::from(pixels.width)).max(2);
            let height = pixel_height.div_ceil(usize::from(pixels.height));
            (pixel_width <= 1024
                && pixel_height <= 1024
                && width <= usize::from(columns)
                && height <= usize::from(rows))
            .then_some(Self {
                columns: width as u16,
                rows: height as u16,
                pixel_width,
                pixel_height,
                grid_columns,
                side,
                gap,
            })
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub columns: usize,
    pub native: Option<GridLayout>,
}

impl Viewport {
    pub fn new(
        columns: u16,
        grid_rows: u16,
        snapshot: Option<&ContextSnapshot>,
        pixels: Option<CellPixels>,
    ) -> Self {
        Self {
            columns: usize::from(columns),
            native: snapshot
                .filter(|snapshot| snapshot.used.is_some() && snapshot.window.is_some())
                .and(pixels)
                .and_then(|pixels| GridLayout::fit(columns, grid_rows, pixels)),
        }
    }

    pub fn first_grid_row(self) -> Option<usize> {
        self.native.map(|grid| {
            if self.columns >= usize::from(grid.columns) + 29 {
                0
            } else {
                3
            }
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
struct RasterKey {
    layout: GridLayout,
    colors: Vec<[u8; 3]>,
    background: [u8; 3],
}

/// Cache one small internal PNG. It never enters the user's attachments or a request.
#[derive(Debug, Default)]
pub struct GridRasterCache {
    entry: Option<(RasterKey, Result<Attachment, String>)>,
}

impl GridRasterCache {
    pub fn get(
        &mut self,
        snapshot: &ContextSnapshot,
        layout: GridLayout,
        base: Style,
        muted: Style,
        background: [u8; 3],
    ) -> Result<&Attachment, String> {
        let cells = snapshot.grid().ok_or("Context grid is unavailable.")?;
        let colors = cells
            .into_iter()
            .map(|cell| match cell.style(base, muted, false).fg {
                Some(Color::Rgb(r, g, b)) => [r, g, b],
                _ => [225; 3],
            })
            .collect();
        let key = RasterKey {
            layout,
            colors,
            background,
        };
        if self.entry.as_ref().is_none_or(|(cached, _)| *cached != key) {
            let image = rounded_grid(&key);
            self.entry = Some((key, image));
        }
        self.entry
            .as_ref()
            .expect("raster was cached")
            .1
            .as_ref()
            .map_err(Clone::clone)
    }
}

fn rounded_grid(key: &RasterKey) -> Result<Attachment, String> {
    let layout = key.layout;
    let mut pixels = vec![key.background; layout.pixel_width * layout.pixel_height];
    let radius = (layout.side / 4).max(2) as f64;
    for (index, color) in key.colors.iter().enumerate() {
        let left = index % layout.grid_columns * (layout.side + layout.gap);
        let top = index / layout.grid_columns * (layout.side + layout.gap);
        for y in 0..layout.side {
            for x in 0..layout.side {
                // Four samples per axis keep a small corner smooth without changing its size.
                let mut covered = 0u16;
                for sy in 0..4 {
                    for sx in 0..4 {
                        let px = x as f64 + (f64::from(sx) + 0.5) / 4.0;
                        let py = y as f64 + (f64::from(sy) + 0.5) / 4.0;
                        let dx = (radius - px)
                            .max(px - (layout.side as f64 - radius))
                            .max(0.0);
                        let dy = (radius - py)
                            .max(py - (layout.side as f64 - radius))
                            .max(0.0);
                        covered += u16::from(dx * dx + dy * dy <= radius * radius);
                    }
                }
                let pixel = &mut pixels[(top + y) * layout.pixel_width + left + x];
                for (channel, (foreground, background)) in
                    pixel.iter_mut().zip(color.iter().zip(key.background))
                {
                    *channel = ((u16::from(*foreground) * covered
                        + u16::from(background) * (16 - covered)
                        + 8)
                        / 16) as u8;
                }
            }
        }
    }
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(
            &mut bytes,
            layout.pixel_width as u32,
            layout.pixel_height as u32,
        );
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|error| error.to_string())?;
        writer
            .write_image_data(pixels.as_flattened())
            .map_err(|error| error.to_string())?;
        writer.finish().map_err(|error| error.to_string())?;
    }
    Attachment::from_image("context-grid.png", &bytes).map_err(|error| error.to_string())
}

impl ContextSnapshot {
    pub fn from_usage_update(update: &Value) -> Self {
        let available = update["_meta"]["aster"]["usageAvailable"] != false;
        // ACP's provider total remains authoritative even if custom metadata
        // is malformed, unsupported or from an older backend.
        let used = update["used"].as_u64().filter(|_| available);
        let metadata = &update["_meta"]["aster"]["context"];
        let supported = metadata["version"].as_u64() == Some(1);
        let request_id = supported
            .then(|| {
                metadata["requestId"]
                    .as_str()
                    .filter(|id| !id.is_empty() && id.len() <= 128)
                    .map(str::to_owned)
            })
            .flatten();
        let can_inspect = request_id.is_some();
        let window = if supported {
            metadata["windowTokens"].as_u64()
        } else {
            update["size"].as_u64()
        }
        .filter(|window| *window > 0);
        let mut snapshot = Self {
            used,
            window,
            reserved: supported
                .then(|| metadata["reservedOutputTokens"].as_u64())
                .flatten(),
            planning_reserve: supported && metadata["outputReserveIsFallback"] == true,
            model: if supported {
                clean(metadata["model"].as_str().unwrap_or(""))
                    .chars()
                    .take(200)
                    .collect()
            } else {
                String::new()
            },
            categories: Vec::new(),
            items: Vec::new(),
            media: supported && metadata["unavailableReason"] == "media",
            content_items: if can_inspect {
                content_items(metadata)
            } else {
                Vec::new()
            },
            output_available: can_inspect && metadata["outputAvailable"] == true,
            request_id,
        };
        if !supported || metadata["attribution"] != "characters" || used.is_none() {
            return snapshot;
        }
        let Some(categories) = metadata["categories"].as_array().filter(|c| c.len() <= 7) else {
            return snapshot;
        };
        let mut parsed: Vec<CategoryUsage> = Vec::new();
        for item in categories {
            let Some(category) = item["id"].as_str().and_then(Category::parse) else {
                return snapshot;
            };
            let Some(tokens) = item["tokens"].as_u64() else {
                return snapshot;
            };
            if parsed.iter().any(|item| item.category == category) {
                return snapshot;
            }
            parsed.push(CategoryUsage { category, tokens });
        }
        if parsed.iter().map(|item| item.tokens as u128).sum::<u128>() != used.unwrap_or(0) as u128
        {
            return snapshot;
        }
        snapshot.categories = parsed;
        if let Some(items) = metadata["items"]
            .as_array()
            .filter(|items| items.len() <= 64)
        {
            for item in items {
                let (Some(category), Some(tokens), Some(key)) = (
                    item["category"].as_str().and_then(Category::parse),
                    item["tokens"].as_u64(),
                    item["key"].as_str(),
                ) else {
                    snapshot.items.clear();
                    return snapshot;
                };
                snapshot.items.push(ItemUsage {
                    category,
                    tokens,
                    key: clean(key).chars().take(120).collect(),
                });
            }
            // Detail rows must agree with every category, not just the total.
            if snapshot.categories.iter().any(|category| {
                snapshot
                    .items
                    .iter()
                    .filter(|item| item.category == category.category)
                    .map(|item| item.tokens as u128)
                    .sum::<u128>()
                    != category.tokens as u128
            }) || snapshot.items.iter().any(|item| {
                !snapshot
                    .categories
                    .iter()
                    .any(|category| category.category == item.category)
            }) {
                snapshot.items.clear();
            }
        }
        snapshot
    }

    pub fn targets(&self, expanded: bool) -> Vec<Target> {
        let mut targets: Vec<_> = [
            Category::SystemPrompt,
            Category::SystemTools,
            Category::MemoryFiles,
            Category::Skills,
            Category::Messages,
            Category::Runtime,
            Category::Other,
        ]
        .into_iter()
        .filter(|category| {
            self.content_items
                .iter()
                .any(|item| item.category == *category)
        })
        .map(Target::Category)
        .collect();
        if self.output_available {
            targets.push(Target::Output);
        }
        if expanded {
            targets.extend(self.content_items.iter().map(|item| Target::Block {
                category: item.category,
                key: item.key.clone(),
            }));
        }
        targets
    }

    fn grid(&self) -> Option<Vec<Cell>> {
        let (Some(used), Some(window)) = (self.used, self.window) else {
            return None;
        };
        let occupied =
            ((used.min(window) as u128 * CELLS as u128).div_ceil(window as u128)) as usize;
        let reserve = self.reserved.unwrap_or(0).min(window.saturating_sub(used));
        let reserved = ((reserve as u128 * CELLS as u128).div_ceil(window as u128) as usize)
            .min(CELLS - occupied);
        let mut cells = Vec::with_capacity(CELLS);
        if self.categories.is_empty() || used == 0 {
            cells.resize(occupied, Cell::Used(None));
        } else {
            let allocations: Vec<_> = self
                .categories
                .iter()
                .map(|item| {
                    let weight = occupied as u128 * item.tokens as u128;
                    (weight / used as u128, weight % used as u128)
                })
                .collect();
            let mut counts: Vec<_> = allocations
                .iter()
                .map(|(count, _)| *count as usize)
                .collect();
            let remainder = occupied - counts.iter().sum::<usize>();
            let mut order: Vec<_> = (0..counts.len()).collect();
            order.sort_by_key(|index| (std::cmp::Reverse(allocations[*index].1), *index));
            for index in order.into_iter().take(remainder) {
                counts[index] += 1;
            }
            for (category, count) in self.categories.iter().zip(counts) {
                cells.extend(std::iter::repeat_n(
                    Cell::Used(Some(category.category)),
                    count,
                ));
            }
        }
        cells.extend(std::iter::repeat_n(Cell::Reserved, reserved));
        cells.resize(CELLS, Cell::Free);
        Some(cells)
    }
}

fn number(tokens: u64) -> String {
    let digits = tokens.to_string();
    let mut result = String::new();
    for (index, digit) in digits.chars().enumerate() {
        if index != 0 && (digits.len() - index).is_multiple_of(3) {
            result.push(',');
        }
        result.push(digit);
    }
    result
}

fn clipped(text: &str, columns: usize) -> String {
    if text.width() <= columns {
        return text.to_owned();
    }
    let mut result = String::new();
    let mut used = 0;
    for character in text.chars() {
        let width = unicode_width::UnicodeWidthChar::width(character).unwrap_or(0);
        if used + width + 1 > columns {
            break;
        }
        result.push(character);
        used += width;
    }
    if columns > 0 {
        result.push('…');
    }
    result
}

fn statistic(
    label: &str,
    tokens: u64,
    approximate: bool,
    window: Option<u64>,
    columns: usize,
    style: Style,
) -> Line<'static> {
    let amount = format!("{}{}", if approximate { "~" } else { "" }, number(tokens));
    let percent = window
        .map(|window| format!(" ({:.1}%)", tokens as f64 / window as f64 * 100.0))
        .unwrap_or_default();
    let label_width = columns
        .saturating_sub(3 + amount.len() + percent.len())
        .min(19);
    Line::from(vec![
        Span::styled(format!("{CELL} "), style),
        Span::styled(
            format!(
                "{:<label_width$} {amount}{percent}",
                clipped(label, label_width),
            ),
            style,
        ),
    ])
}

/// Reserve the native grid footprint, or draw compact character cells as a fallback.
pub fn lines(
    snapshot: Option<&ContextSnapshot>,
    model: &str,
    browser: Browser,
    viewport: Viewport,
    monochrome: bool,
    base: Style,
    muted: Style,
) -> Rows {
    let columns = viewport.columns;
    let mut rows = Vec::new();
    let mut hit_rows = Vec::new();
    let heading = base.add_modifier(Modifier::BOLD);
    let reserved_style = Cell::Reserved.style(base, muted, monochrome);
    let Some(snapshot) = snapshot else {
        rows.push(Line::styled("Awaiting the first model response.", base));
        rows.push(Line::styled(
            "The grid shows the latest completed model request.",
            muted,
        ));
        return Rows {
            lines: rows,
            targets: hit_rows,
        };
    };
    let targets = snapshot.targets(browser.expanded);
    let model = if snapshot.model.is_empty() {
        model.to_owned()
    } else {
        snapshot.model.clone()
    };
    let summary = match (snapshot.used, snapshot.window) {
        (Some(used), Some(window)) => {
            format!("{} / {} tokens", number(used), number(window))
        }
        (Some(used), None) => format!("{} tokens · limit unknown", number(used)),
        (None, _) => "Input-token usage unavailable".into(),
    };
    if let Some(cells) = snapshot.grid() {
        let grid_columns = viewport.native.map_or_else(
            || {
                [20, 10, 5, 2, 1]
                    .into_iter()
                    .find(|count| *count <= columns)
                    .unwrap_or(1)
            },
            |grid| grid.grid_columns,
        );
        let grid_width = viewport
            .native
            .map_or(grid_columns, |grid| usize::from(grid.columns));
        let alongside = columns >= grid_width + 29;
        let per_cell = format!(
            "Each cell ≈ {} tokens",
            number(snapshot.window.unwrap_or(1).div_ceil(CELLS as u64))
        );
        let percent = format!(
            "{:.1}% of total window",
            snapshot.used.unwrap_or(0) as f64 / snapshot.window.unwrap_or(1) as f64 * 100.0
        );
        let notes = [
            model.clone(),
            summary.clone(),
            percent,
            per_cell.clone(),
            "Latest completed request".into(),
        ];
        if !alongside {
            rows.push(Line::styled(clipped(&model, columns), heading));
            rows.push(Line::styled(clipped(&summary, columns), base));
            rows.push(Line::styled("", base));
        }
        let grid_rows = viewport
            .native
            .map_or(CELLS / grid_columns, |grid| usize::from(grid.rows));
        for index in 0..grid_rows.max(if alongside { notes.len() } else { 0 }) {
            let mut spans = if viewport.native.is_some() {
                vec![Span::styled(" ".repeat(grid_width), base)]
            } else {
                cells[index * grid_columns..(index + 1) * grid_columns]
                    .iter()
                    .map(|cell| Span::styled(CELL, cell.style(base, muted, monochrome)))
                    .collect()
            };
            if alongside {
                if let Some(note) = notes.get(index) {
                    spans.push(Span::styled("   ", base));
                    spans.push(Span::styled(
                        clipped(note, columns.saturating_sub(grid_width + 3)),
                        if index < 2 { heading } else { muted },
                    ));
                }
            }
            rows.push(Line::from(spans));
        }
        if !alongside {
            rows.push(Line::styled(per_cell, muted));
        }
    } else {
        rows.push(Line::styled(model, heading));
        rows.push(Line::styled(summary, base));
        rows.push(Line::styled(
            if snapshot.used.is_none() {
                "The provider did not report input-token usage."
            } else {
                "The grid needs a known model context limit."
            },
            muted,
        ));
    }
    rows.push(Line::styled("", base));
    rows.push(Line::styled("Usage by category", heading));
    for category in &snapshot.categories {
        selectable(
            statistic(
                category.category.label(),
                category.tokens,
                true,
                snapshot.window,
                columns,
                category.category.style(base, monochrome),
            ),
            targets
                .iter()
                .position(|target| *target == Target::Category(category.category)),
            browser,
            &mut rows,
            &mut hit_rows,
        );
    }
    if snapshot.categories.is_empty() {
        if let Some(used) = snapshot.used {
            rows.push(statistic(
                "Used context",
                used,
                false,
                snapshot.window,
                columns,
                base,
            ));
        }
        rows.push(Line::styled(
            if snapshot.media {
                "Category split unavailable for image/audio/file input."
            } else {
                "Category split unavailable."
            },
            muted,
        ));
    }
    for target in &targets {
        if let Target::Category(category) = target {
            if !snapshot
                .categories
                .iter()
                .any(|usage| usage.category == *category)
            {
                selectable(
                    Line::styled(
                        format!("{CELL} {}   Enter view", category.label()),
                        category.style(base, monochrome),
                    ),
                    targets.iter().position(|candidate| candidate == target),
                    browser,
                    &mut rows,
                    &mut hit_rows,
                );
            }
        }
    }
    if let Some(reserve) = snapshot.reserved.filter(|reserve| *reserve > 0) {
        rows.push(statistic(
            if snapshot.planning_reserve {
                "Planning reserve"
            } else {
                "Output reserve"
            },
            reserve,
            false,
            snapshot.window,
            columns,
            reserved_style,
        ));
    }
    if let (Some(used), Some(window)) = (snapshot.used, snapshot.window) {
        let free = window.saturating_sub(used.saturating_add(snapshot.reserved.unwrap_or(0)));
        rows.push(statistic(
            "Free space",
            free,
            false,
            Some(window),
            columns,
            muted,
        ));
        let excess = used.saturating_sub(window.saturating_sub(snapshot.reserved.unwrap_or(0)));
        if excess > 0 {
            rows.push(Line::styled(
                format!("Over usable input budget by {} tokens.", number(excess)),
                if monochrome {
                    heading
                } else {
                    heading.fg(Color::Rgb(224, 130, 130))
                },
            ));
        }
    }
    rows.push(Line::styled("", base));
    rows.push(Line::styled(
        "Total from provider; ~ categories attributed by text size.",
        muted,
    ));
    rows.push(Line::styled(
        "Input grid excludes drafts and this request's output.",
        muted,
    ));
    if snapshot.output_available {
        rows.push(Line::styled("", base));
        selectable(
            Line::styled(format!("{CELL} Latest model output   Enter view"), base),
            targets.iter().position(|target| *target == Target::Output),
            browser,
            &mut rows,
            &mut hit_rows,
        );
    }
    if browser.expanded {
        rows.push(Line::styled("", base));
        rows.push(Line::styled("Context blocks and messages", heading));
        for item in &snapshot.content_items {
            let amount = snapshot
                .items
                .iter()
                .find(|usage| usage.category == item.category && usage.key == clean(&item.key))
                .map_or_else(
                    || format!("{} chars", number(item.characters)),
                    |usage| format!("~{}", number(usage.tokens)),
                );
            let target = Target::Block {
                category: item.category,
                key: item.key.clone(),
            };
            selectable(
                Line::styled(
                    format!(
                        "{CELL} {}  {amount}",
                        clipped(&clean(&item.key), columns.saturating_sub(amount.len() + 4))
                    ),
                    item.category.style(base, monochrome),
                ),
                targets.iter().position(|candidate| *candidate == target),
                browser,
                &mut rows,
                &mut hit_rows,
            );
        }
        for item in snapshot
            .items
            .iter()
            .filter(|_| snapshot.content_items.is_empty())
        {
            let amount = format!("~{}", number(item.tokens));
            rows.push(Line::styled(
                format!(
                    "{CELL} {}  {amount}",
                    clipped(&item.key, columns.saturating_sub(amount.len() + 4)),
                ),
                item.category.style(base, monochrome),
            ));
        }
        if snapshot.items.is_empty() && snapshot.content_items.is_empty() {
            rows.push(Line::styled("No per-item attribution available.", muted));
        }
    }
    Rows {
        lines: rows,
        targets: hit_rows,
    }
}

fn selectable(
    mut line: Line<'static>,
    target: Option<usize>,
    browser: Browser,
    rows: &mut Vec<Line<'static>>,
    hits: &mut Vec<(usize, usize)>,
) {
    if let Some(index) = target {
        hits.push((rows.len(), index));
        if index == browser.selected {
            let style = line
                .spans
                .first()
                .map_or(line.style, |span| line.style.patch(span.style));
            line = line.patch_style(style.add_modifier(Modifier::REVERSED));
        }
    }
    rows.push(line);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn snapshot(used: u64, window: u64, reserve: u64) -> ContextSnapshot {
        ContextSnapshot::from_usage_update(&json!({
            "used": used,
            "size": window,
            "_meta": {"aster": {"context": {
                "version": 1, "windowTokens": window,
                "reservedOutputTokens": reserve, "attribution": "characters",
                "categories": [
                    {"id": "system_prompt", "tokens": used / 4},
                    {"id": "messages", "tokens": used - used / 4}
                ],
                "items": [
                    {"category": "system_prompt", "key": "system_prompt", "tokens": used / 4},
                    {"category": "messages", "key": "Conversation text", "tokens": used - used / 4}
                ]
            }}}
        }))
    }

    #[test]
    fn cells_preserve_capacity_and_show_small_nonzero_usage() {
        for (used, window, reserve) in [
            (0, 200_000, 10_000),
            (1, 200_000, 10_000),
            (40_000, 200_000, 10_000),
            (195_000, 200_000, 10_000),
            (300_000, 200_000, 10_000),
            (u64::MAX, u64::MAX, u64::MAX),
        ] {
            let grid = snapshot(used, window, reserve).grid().unwrap();
            assert_eq!(grid.len(), CELLS);
            assert_eq!(
                grid.iter()
                    .filter(|cell| matches!(cell, Cell::Used(_)))
                    .count(),
                ((used.min(window) as u128 * CELLS as u128).div_ceil(window as u128)) as usize,
            );
        }
    }

    #[test]
    fn categories_and_reserve_have_distinct_cells() {
        let grid = snapshot(40_000, 200_000, 10_000).grid().unwrap();
        assert_eq!(
            grid.iter()
                .filter(|cell| **cell == Cell::Used(Some(Category::SystemPrompt)))
                .count(),
            5
        );
        assert_eq!(
            grid.iter()
                .filter(|cell| **cell == Cell::Used(Some(Category::Messages)))
                .count(),
            15
        );
        assert_eq!(
            grid.iter().filter(|cell| **cell == Cell::Reserved).count(),
            5
        );
        assert_eq!(grid.iter().filter(|cell| **cell == Cell::Free).count(), 75);
    }

    #[test]
    fn native_grid_has_rounded_square_cells_and_two_pixel_gaps_on_both_axes() {
        let layout = GridLayout::fit(68, 18, CellPixels::new(10, 23).unwrap()).unwrap();
        assert_eq!((layout.pixel_width, layout.pixel_height), (318, 78));
        assert_eq!((layout.columns, layout.rows), (32, 4));
        assert_eq!((layout.side, layout.gap), (14, 2));
        let mut cache = GridRasterCache::default();
        let snapshot = snapshot(40_000, 200_000, 10_000);
        let Attachment::Image { data, .. } = cache
            .get(
                &snapshot,
                layout,
                Style::default(),
                Style::default(),
                [32; 3],
            )
            .unwrap()
        else {
            panic!("grid must be an internal PNG")
        };
        let raster = crate::image_preview::decode_png_pixels(data, 318, 78, [32; 3]).unwrap();
        assert_eq!((raster.width, raster.height), (318, 78));
        let at = |x, y| raster.pixels[y * raster.width + x];
        assert_eq!(at(0, 0), [32; 3], "corner is rounded away");
        assert_eq!(at(7, 0), [184, 157, 205], "flat top edge");
        assert_eq!(at(0, 7), [184, 157, 205], "flat left edge");
        assert!(at(1, 0)[0] > 32 && at(1, 0)[0] < 184, "antialiased corner");
        for index in 0..100 {
            let center = at(index % 20 * 16 + 7, index / 20 * 16 + 7);
            let expected = match index {
                0..=4 => [184, 157, 205],
                5..=19 => [147, 186, 133],
                20..=24 => [156, 168, 184],
                _ => [98; 3],
            };
            assert_eq!(center, expected, "cell {index}");
        }
        for y in 0..78 {
            assert_eq!(at(14, y), [32; 3]);
            assert_eq!(at(15, y), [32; 3]);
        }
        for x in 0..318 {
            assert_eq!(at(x, 14), [32; 3]);
            assert_eq!(at(x, 15), [32; 3]);
        }
    }

    #[test]
    fn native_grid_fits_complete_footprints_and_reuses_unchanged_cell_colors() {
        for pixels in [
            CellPixels::new(1, 1).unwrap(),
            CellPixels::new(10, 23).unwrap(),
            CellPixels::new(20, 48).unwrap(),
            CellPixels::new(512, 512).unwrap(),
        ] {
            for (columns, rows) in [(1, 1), (4, 2), (12, 20), (34, 26), (68, 18)] {
                if let Some(layout) = GridLayout::fit(columns, rows, pixels) {
                    assert!(layout.columns <= columns && layout.rows <= rows);
                    assert!(layout.pixel_width <= 1024 && layout.pixel_height <= 1024);
                    assert!(layout.pixel_width <= usize::from(columns) * usize::from(pixels.width));
                    assert!(layout.pixel_height <= usize::from(rows) * usize::from(pixels.height));
                }
            }
        }
        assert!(GridLayout::fit(
            68,
            18,
            CellPixels {
                width: 0,
                height: 23
            }
        )
        .is_none());
        assert!(GridLayout::fit(68, 0, CellPixels::new(10, 23).unwrap()).is_none());
        let layout = GridLayout::fit(68, 18, CellPixels::new(10, 23).unwrap()).unwrap();
        let mut cache = GridRasterCache::default();
        let mut snapshot = snapshot(40_000, 200_000, 10_000);
        let style = Style::default();
        let Attachment::Image { data, .. } =
            cache.get(&snapshot, layout, style, style, [32; 3]).unwrap()
        else {
            panic!("expected internal PNG")
        };
        let original = data.as_ptr();
        snapshot.model = "a different model with the same composition".into();
        let Attachment::Image { data, .. } =
            cache.get(&snapshot, layout, style, style, [32; 3]).unwrap()
        else {
            panic!("expected internal PNG")
        };
        assert_eq!(original, data.as_ptr(), "timer redraws reuse the raster");
        snapshot.used = Some(100_000);
        snapshot.categories.clear();
        let Attachment::Image { data, .. } =
            cache.get(&snapshot, layout, style, style, [32; 3]).unwrap()
        else {
            panic!("expected internal PNG")
        };
        assert_ne!(original, data.as_ptr(), "changed colors replace the raster");
    }

    #[test]
    fn unknown_usage_or_capacity_does_not_draw_an_empty_grid() {
        for update in [
            json!({"used": 400, "size": 0}),
            json!({"used": 0, "size": 200_000, "_meta": {"aster": {"usageAvailable": false}}}),
            json!({"used": 400, "size": 200_000, "_meta": {"aster": {"context": {
                "version": 1, "windowTokens": null
            }}}}),
        ] {
            assert!(ContextSnapshot::from_usage_update(&update).grid().is_none());
        }
    }

    #[test]
    fn inconsistent_attribution_cannot_replace_the_provider_total() {
        let value = json!({"used": 40, "size": 100, "_meta": {"aster": {"context": {
            "version": 1, "windowTokens": 100, "attribution": "characters",
            "categories": [{"id": "messages", "tokens": 50}]
        }}}});
        let snapshot = ContextSnapshot::from_usage_update(&value);
        assert_eq!(snapshot.used, Some(40));
        assert!(snapshot.categories.is_empty());
        assert_eq!(
            snapshot
                .grid()
                .unwrap()
                .iter()
                .filter(|cell| matches!(cell, Cell::Used(_)))
                .count(),
            40
        );
    }

    #[test]
    fn unsupported_or_malformed_metadata_keeps_the_standard_usage() {
        for metadata in [
            json!({"version": 2, "windowTokens": 0}),
            json!({"version": 1, "windowTokens": 100, "attribution": "characters",
                "categories": [{"id": "messages", "tokens": 40}, {"id": "messages", "tokens": 0}]}),
            json!({"version": 1, "windowTokens": 100, "attribution": "characters",
                "categories": [{"id": "unknown", "tokens": 40}]}),
            json!({"version": 1, "windowTokens": 100, "attribution": "characters",
                "categories": [{"id": "messages", "tokens": u64::MAX},
                    {"id": "system_prompt", "tokens": u64::MAX}]}),
        ] {
            let value = json!({"used": 40, "size": 100,
                "_meta": {"aster": {"context": metadata}}});
            let snapshot = ContextSnapshot::from_usage_update(&value);
            assert_eq!(snapshot.used, Some(40));
            assert_eq!(snapshot.window, Some(100));
            assert!(snapshot.categories.is_empty());
            assert_eq!(snapshot.grid().unwrap().len(), CELLS);
        }
    }

    #[test]
    fn inconsistent_details_do_not_hide_valid_category_totals() {
        let value = json!({"used": 40, "size": 100, "_meta": {"aster": {"context": {
            "version": 1, "windowTokens": 100, "attribution": "characters",
            "categories": [{"id": "messages", "tokens": 40}],
            "items": [{"category": "messages", "tokens": 50, "key": "wrong"}]
        }}}});
        let snapshot = ContextSnapshot::from_usage_update(&value);
        assert_eq!(snapshot.categories[0].tokens, 40);
        assert!(snapshot.items.is_empty());
    }
}
