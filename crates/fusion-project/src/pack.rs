//! Portable, declarative material packs. Installing one copies definitions
//! into the scene, so moving the scene never loses its materials.

use crate::{MaterialEntry, Project, Representation};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// A versioned library of material definitions; contains no executable code.
pub struct MaterialPack {
    /// Pack schema version, currently one.
    pub format: u32,
    /// Human-readable library name.
    pub name: String,
    /// Publisher-assigned library revision.
    pub version: String,
    /// Description shown before installation.
    #[serde(default)]
    pub description: String,
    /// Materials copied into the destination project.
    #[serde(rename = "material")]
    pub materials: Vec<MaterialEntry>,
}

impl MaterialPack {
    /// Capture all of a project's materials as a new library.
    pub fn from_project(project: &Project, name: &str) -> Self {
        Self {
            format: 1,
            name: name.into(),
            version: "1.0.0".into(),
            description: String::new(),
            materials: project.materials.clone(),
        }
    }

    /// Parse and validate a pack before presenting or installing it.
    pub fn from_toml(text: &str) -> Result<Self, String> {
        let pack: Self = toml::from_str(text).map_err(|e| format!("Invalid material pack: {e}"))?;
        pack.validate()?;
        Ok(pack)
    }

    /// Serialize a valid pack to portable TOML.
    pub fn to_toml(&self) -> Result<String, String> {
        self.validate()?;
        toml::to_string(self).map_err(|e| e.to_string())
    }

    /// Read a pack from disk.
    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        Self::from_toml(&std::fs::read_to_string(path).map_err(|e| e.to_string())?)
    }

    /// Whether any material uses the continuum solver.
    pub fn requires_advanced(&self) -> bool {
        self.materials
            .iter()
            .any(|m| m.representation == Representation::Deformable)
    }

    /// Check names, format and physical/visual parameter ranges.
    pub fn validate(&self) -> Result<(), String> {
        if self.format != 1 {
            return Err("Unsupported material pack format".into());
        }
        if self.name.trim().is_empty() || self.version.trim().is_empty() {
            return Err("A pack needs a name and version".into());
        }
        if self.materials.is_empty() {
            return Err("The pack has no materials".into());
        }
        let mut names = std::collections::HashSet::new();
        for material in &self.materials {
            if material.name.trim().is_empty() || !names.insert(&material.name) {
                return Err(format!(
                    "Empty or duplicate material name: {}",
                    material.name
                ));
            }
            material.validate()?;
        }
        Ok(())
    }

    /// Transactional merge. Identical definitions are deduplicated; a name
    /// collision with different physics is refused before any mutation.
    pub fn install_into(&self, project: &mut Project) -> Result<usize, String> {
        self.validate()?;
        let mut additions = Vec::new();
        for material in &self.materials {
            match project.materials.iter().find(|m| m.name == material.name) {
                Some(existing) if existing != material => return Err(format!(
                    "Material '{}' already exists with different properties. Rename it or the pack entry first.", material.name)),
                Some(_) => {},
                None => additions.push(material.clone()),
            }
        }
        let count = additions.len();
        if count > 0 {
            project.materials.extend(additions);
            project.state = None;
            let dependency = (format!("material-pack:{}", self.name), self.version.clone());
            project
                .metadata
                .tags
                .retain(|(key, _)| key != &dependency.0);
            project.metadata.tags.push(dependency);
        }
        Ok(count)
    }
}
