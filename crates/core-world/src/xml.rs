//! Shared zero-dependency XML tree parser and pose math helpers for
//! URDF and MJCF importers.

use std::collections::HashMap;

use core_math::{Pose, Vec3};

#[derive(Clone, Debug)]
pub(crate) struct XmlNode {
    pub(crate) tag: String,
    pub(crate) attrs: HashMap<String, String>,
    pub(crate) children: Vec<XmlNode>,
}

/// Parses all top-level XML element nodes in `input`.
pub(crate) fn parse_xml_roots(input: &str) -> Result<Vec<XmlNode>, String> {
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut stack: Vec<XmlNode> = Vec::new();
    let mut roots: Vec<XmlNode> = Vec::new();

    while i < bytes.len() {
        while i < bytes.len() && bytes[i] != b'<' {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        if input[i..].starts_with("<!--") {
            let Some(end) = input[i + 4..].find("-->") else {
                return Err("unterminated XML comment".into());
            };
            i += 4 + end + 3;
            continue;
        }
        if input[i..].starts_with("<?") {
            let Some(end) = input[i + 2..].find("?>") else {
                return Err("unterminated XML processing instruction".into());
            };
            i += 2 + end + 2;
            continue;
        }
        if input[i..].starts_with("<![CDATA[") {
            let Some(end) = input[i + 9..].find("]]>") else {
                return Err("unterminated XML CDATA section".into());
            };
            i += 9 + end + 3;
            continue;
        }
        if input[i..].starts_with("<!") {
            i += 2;
            let mut brackets = 0usize;
            let mut quote: Option<u8> = None;
            while i < bytes.len() {
                let b = bytes[i];
                if let Some(q) = quote {
                    if b == q {
                        quote = None;
                    }
                } else {
                    match b {
                        b'"' | b'\'' => quote = Some(b),
                        b'[' => brackets += 1,
                        b']' => brackets = brackets.saturating_sub(1),
                        b'>' if brackets == 0 => {
                            i += 1;
                            break;
                        }
                        _ => {}
                    }
                }
                i += 1;
            }
            if quote.is_some() || brackets != 0 {
                return Err("unterminated XML declaration".into());
            }
            continue;
        }
        if input[i..].starts_with("</") {
            i += 2;
            let start = i;
            while i < bytes.len() && bytes[i] != b'>' {
                i += 1;
            }
            if i >= bytes.len() {
                return Err("unterminated closing XML tag".into());
            }
            let close_name = input[start..i].trim();
            i += 1;
            let node = stack
                .pop()
                .ok_or_else(|| format!("unexpected closing tag </{close_name}>"))?;
            if node.tag != close_name {
                return Err(format!(
                    "mismatched closing tag </{close_name}> for <{}>",
                    node.tag
                ));
            }
            if let Some(parent) = stack.last_mut() {
                parent.children.push(node);
            } else {
                roots.push(node);
            }
            continue;
        }

        // Opening or self-closing tag.
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let name_start = i;
        while i < bytes.len()
            && !bytes[i].is_ascii_whitespace()
            && bytes[i] != b'>'
            && bytes[i] != b'/'
        {
            i += 1;
        }
        let tag = input[name_start..i].to_string();
        if tag.is_empty() {
            return Err("empty XML tag name".into());
        }
        let mut attrs = HashMap::new();
        let mut self_closing = false;
        loop {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(format!("unterminated tag <{tag}>"));
            }
            if bytes[i] == b'/' {
                self_closing = true;
                i += 1;
                if i >= bytes.len() || bytes[i] != b'>' {
                    return Err(format!("expected '>' after '/' in <{tag}>"));
                }
                i += 1;
                break;
            }
            if bytes[i] == b'>' {
                i += 1;
                break;
            }
            let k_start = i;
            while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            let key = input[k_start..i].to_string();
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] != b'=' {
                return Err(format!("expected '=' after attribute {key} in <{tag}>"));
            }
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() || (bytes[i] != b'"' && bytes[i] != b'\'') {
                return Err(format!("expected quoted attribute value for {key}"));
            }
            let quote = bytes[i];
            i += 1;
            let v_start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            if i >= bytes.len() {
                return Err(format!("unterminated quoted value for {key}"));
            }
            let val = input[v_start..i].to_string();
            i += 1;
            attrs.insert(key, val);
        }
        let node = XmlNode {
            tag,
            attrs,
            children: Vec::new(),
        };
        if self_closing {
            if let Some(parent) = stack.last_mut() {
                parent.children.push(node);
            } else {
                roots.push(node);
            }
        } else {
            stack.push(node);
        }
    }

    if !stack.is_empty() {
        return Err(format!("unclosed XML tag <{}>", stack.last().unwrap().tag));
    }
    Ok(roots)
}

/// Parses XML and returns the top-level element matching `expected_root`.
pub(crate) fn parse_xml_doc(input: &str, expected_root: &str) -> Result<XmlNode, String> {
    parse_xml_roots(input)?
        .into_iter()
        .find(|n| n.tag == expected_root)
        .ok_or_else(|| format!("missing <{expected_root}> root element"))
}

pub(crate) fn parse_f64_list(s: &str) -> Result<Vec<f64>, String> {
    s.split_whitespace()
        .map(|w| {
            w.parse::<f64>()
                .map_err(|_| format!("invalid float '{w}' in '{s}'"))
        })
        .collect()
}

pub(crate) fn parse_vec3(s: &str) -> Result<Vec3, String> {
    let parts = parse_f64_list(s)?;
    if parts.len() != 3 {
        return Err(format!("expected 3 floats in '{s}'"));
    }
    Ok(Vec3::new(parts[0], parts[1], parts[2]))
}

pub(crate) fn compose_pose(a: Pose, b: Pose) -> Pose {
    Pose::new(
        a.position + a.rotation.rotate(b.position),
        (a.rotation * b.rotation).normalize(),
    )
}

pub(crate) fn inverse_pose(a: Pose) -> Pose {
    let inv_q = a.rotation.conjugate();
    Pose::new(inv_q.rotate(-a.position), inv_q)
}
