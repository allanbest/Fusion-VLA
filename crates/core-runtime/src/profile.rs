//! Where the time goes, by name: measured, never assumed.

use std::fmt::Write;
use std::time::{Duration, Instant};

/// Accumulated time and calls per named section.
#[derive(Clone, Debug)]
pub struct Profile {
    enabled: bool,
    entries: Vec<Entry>,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            enabled: true,
            entries: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: &'static str,
    pub total: Duration,
    pub calls: u64,
}

impl Profile {
    /// Enable section timing. Disabled sections execute without clock reads
    /// or entry searches; existing totals are retained until `clear`.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Accumulate another solve without losing its section call counts.
    pub fn merge(&mut self, other: &Self) {
        if !self.enabled {
            return;
        }
        for source in &other.entries {
            if let Some(entry) = self.entries.iter_mut().find(|e| e.name == source.name) {
                entry.total += source.total;
                entry.calls += source.calls;
            } else {
                self.entries.push(source.clone());
            }
        }
    }

    /// Runs `f`, adding its duration to `name`.
    pub fn time<R>(&mut self, name: &'static str, f: impl FnOnce() -> R) -> R {
        if !self.enabled {
            return f();
        }
        let start = Instant::now();
        let result = f();
        self.add(name, start.elapsed());
        result
    }

    pub fn add(&mut self, name: &'static str, elapsed: Duration) {
        if !self.enabled {
            return;
        }
        match self.entries.iter_mut().find(|e| e.name == name) {
            Some(entry) => {
                entry.total += elapsed;
                entry.calls += 1;
            }
            None => self.entries.push(Entry {
                name,
                total: elapsed,
                calls: 1,
            }),
        }
    }

    /// Entries in the order they were first seen.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// One line per entry, largest first, with its share of the total.
    pub fn report(&self) -> String {
        let total: Duration = self.entries.iter().map(|e| e.total).sum();
        let mut sorted: Vec<&Entry> = self.entries.iter().collect();
        sorted.sort_by_key(|e| std::cmp::Reverse(e.total));
        let mut out = String::new();
        for e in sorted {
            let share = 100.0 * e.total.as_secs_f64() / total.as_secs_f64().max(1e-12);
            let _ = writeln!(
                out,
                "{:>24}  {:>9.3} ms  {:>5.1}%  {:>7} calls",
                e.name,
                e.total.as_secs_f64() * 1e3,
                share,
                e.calls
            );
        }
        out
    }
}
