//! The results store: where a run's numbers go so they can be found again.
//!
//! §23 calls the research workflow "mostly orchestration over the project
//! format plus a results store", and this is the second half. It is
//! deliberately a directory of files rather than a database: a study is
//! something you come back to in six months, hand to somebody else, or check
//! into a repository next to the paper, and none of those survive a format
//! only one program can read.
//!
//! Each study is a directory holding `study.toml` — the records — and the
//! `.fusion` scene of every run that was asked to keep one. **A row in the
//! table and the scene that produced it, side by side.** That pairing is the
//! whole point: a result you cannot re-run is an anecdote.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::sweep::SweepResults;

/// Anything that can go wrong writing or reading a study.
#[derive(Debug)]
pub enum StoreError {
    /// The filesystem said no.
    Io(std::io::Error),
    /// The study file is not valid TOML, or not a study.
    Parse(toml::de::Error),
    /// The study could not be turned into TOML.
    Serialize(toml::ser::Error),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Parse(e) => write!(f, "{e}"),
            Self::Serialize(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StoreError {}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// One study on disk: what was run, when, and what came out.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Study {
    /// What this study was for, in the words of whoever ran it.
    pub question: String,
    /// Which Fusion produced it.
    pub fusion_version: String,
    /// When, as an RFC-3339-ish stamp. A string rather than a timestamp type,
    /// because the only thing anybody does with it is read it.
    pub recorded: String,
    /// The results.
    pub results: SweepResults,
}

impl Study {
    /// A study of these results, asking this question.
    pub fn new(question: impl Into<String>, results: SweepResults) -> Self {
        Self {
            question: question.into(),
            fusion_version: env!("CARGO_PKG_VERSION").to_string(),
            recorded: stamp(),
            results,
        }
    }

    /// Write the study to a directory, creating it if needed.
    ///
    /// Writes `study.toml` and `results.csv` — the second because the first
    /// thing anybody does with a table is put it in something else, and a
    /// study nobody can open in a spreadsheet gets retyped by hand.
    pub fn save(&self, dir: impl AsRef<Path>) -> Result<PathBuf, StoreError> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let text = toml::to_string_pretty(self).map_err(StoreError::Serialize)?;
        let path = dir.join("study.toml");
        std::fs::write(&path, text)?;
        std::fs::write(dir.join("results.csv"), self.results.to_csv())?;
        Ok(path)
    }

    /// Read a study back.
    pub fn load(dir: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = dir.as_ref().join("study.toml");
        let text = std::fs::read_to_string(path)?;
        toml::from_str(&text).map_err(StoreError::Parse)
    }
}

/// A coarse timestamp, without taking a dependency on a date library for one
/// string. Seconds since the epoch, which sorts correctly and is unambiguous
/// about time zones by not having one.
fn stamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("unix:{secs}")
}
