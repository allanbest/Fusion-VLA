//! Source text to tokens, including the indentation Python makes structural.
//!
//! ## Why the indentation is done here
//!
//! Python has no braces: a block is a run of lines indented further than the
//! line that opened it. That is a *lexical* fact — it is decided by counting
//! spaces at the start of a line — and a parser that tries to decide it while
//! also deciding what an expression is ends up doing both badly. So the
//! scanner emits [`Token::Indent`] and [`Token::Dedent`] where the level
//! changes, and the parser sees a token stream with explicit block delimiters,
//! exactly as if the language had braces after all.
//!
//! ## Tabs
//!
//! A tab counts as one column, not four or eight. Any choice here is arbitrary
//! and all of them are wrong for somebody, so this takes the one that cannot
//! silently disagree with an editor's rendering: mixing tabs and spaces at the
//! same nesting level is rejected outright rather than resolved.

use crate::ScriptError;

/// One lexical item.
#[derive(Clone, Debug, PartialEq)]
pub enum Token {
    /// A name, or a keyword — the parser decides which.
    Name(String),
    Number(f32),
    /// `+`, `==`, `:`, and the rest. Interned as a `&'static str` so the
    /// parser can compare with `==` against a literal.
    Symbol(&'static str),
    /// End of a logical line.
    Newline,
    /// A block opened.
    Indent,
    /// A block closed.
    Dedent,
    /// End of input.
    End,
}

/// A token and the line it came from, for error messages that point somewhere.
#[derive(Clone, Debug, PartialEq)]
pub struct Spanned {
    pub token: Token,
    pub line: u32,
}

/// Operators and punctuation, longest first so that `<=` is not read as `<`
/// followed by `=`.
const SYMBOLS: &[&str] = &[
    "**", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "//", "(", ")", "[", "]", ",", ":", ".",
    "+", "-", "*", "/", "%", "<", ">", "=",
];

/// Turn `source` into tokens.
pub fn scan(source: &str) -> Result<Vec<Spanned>, ScriptError> {
    let mut out = Vec::new();
    let mut indents: Vec<usize> = vec![0];
    let mut line_number = 0u32;
    // Bracket depth. Inside brackets a newline is whitespace, which is what
    // lets a long call be split across lines.
    let mut depth = 0i32;

    for raw in source.lines() {
        line_number += 1;

        if depth == 0 {
            let (level, rest) = measure_indent(raw, line_number)?;
            // Blank and comment-only lines carry no structure. Reading their
            // indentation would close a block every time somebody left a blank
            // line in the middle of one.
            if rest.trim().is_empty() {
                continue;
            }
            match level.cmp(indents.last().expect("never empty")) {
                std::cmp::Ordering::Greater => {
                    indents.push(level);
                    out.push(Spanned {
                        token: Token::Indent,
                        line: line_number,
                    });
                }
                std::cmp::Ordering::Less => {
                    while *indents.last().expect("never empty") > level {
                        indents.pop();
                        out.push(Spanned {
                            token: Token::Dedent,
                            line: line_number,
                        });
                    }
                    if *indents.last().expect("never empty") != level {
                        return Err(ScriptError::new(
                            line_number,
                            "this line does not line up with any block it could belong to",
                        ));
                    }
                }
                std::cmp::Ordering::Equal => {}
            }
        }

        let before = out.len();
        scan_line(raw, line_number, &mut depth, &mut out)?;
        // A line that produced nothing but a comment has no logical end.
        if out.len() > before && depth == 0 {
            out.push(Spanned {
                token: Token::Newline,
                line: line_number,
            });
        }
    }

    if depth != 0 {
        return Err(ScriptError::new(line_number, "a bracket is never closed"));
    }
    while indents.len() > 1 {
        indents.pop();
        out.push(Spanned {
            token: Token::Dedent,
            line: line_number,
        });
    }
    out.push(Spanned {
        token: Token::End,
        line: line_number,
    });
    Ok(out)
}

/// Leading whitespace as a column count, and the rest of the line.
fn measure_indent(line: &str, number: u32) -> Result<(usize, &str), ScriptError> {
    let mut columns = 0usize;
    let mut spaces = false;
    let mut tabs = false;
    for (index, ch) in line.char_indices() {
        match ch {
            ' ' => {
                spaces = true;
                columns += 1;
            }
            '\t' => {
                tabs = true;
                columns += 1;
            }
            _ => {
                if spaces && tabs {
                    return Err(ScriptError::new(
                        number,
                        "this line is indented with both tabs and spaces, and there is no \
                         reading of that which is not a guess",
                    ));
                }
                return Ok((columns, &line[index..]));
            }
        }
    }
    Ok((columns, ""))
}

fn scan_line(
    line: &str,
    number: u32,
    depth: &mut i32,
    out: &mut Vec<Spanned>,
) -> Result<(), ScriptError> {
    let bytes = line.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        let c = bytes[i];
        if c == b' ' || c == b'\t' || c == b'\r' {
            i += 1;
            continue;
        }
        // A comment ends the line, whatever is left of it.
        if c == b'#' {
            break;
        }

        if c.is_ascii_digit() || (c == b'.' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit)) {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_digit()
                    || bytes[i] == b'.'
                    || matches!(bytes[i], b'e' | b'E')
                    || ((bytes[i] == b'+' || bytes[i] == b'-')
                        && matches!(bytes[i - 1], b'e' | b'E')))
            {
                i += 1;
            }
            let text = &line[start..i];
            let value: f32 = text
                .parse()
                .map_err(|_| ScriptError::new(number, format!("{text} is not a number")))?;
            out.push(Spanned {
                token: Token::Number(value),
                line: number,
            });
            continue;
        }

        if c == b'_' || c.is_ascii_alphabetic() {
            let start = i;
            while i < bytes.len() && (bytes[i] == b'_' || bytes[i].is_ascii_alphanumeric()) {
                i += 1;
            }
            out.push(Spanned {
                token: Token::Name(line[start..i].to_string()),
                line: number,
            });
            continue;
        }

        let rest = &line[i..];
        let Some(symbol) = SYMBOLS.iter().find(|s| rest.starts_with(**s)) else {
            let ch = rest.chars().next().unwrap_or('?');
            return Err(ScriptError::new(
                number,
                format!("{ch:?} is not something this language has a meaning for"),
            ));
        };
        match *symbol {
            "(" | "[" => *depth += 1,
            ")" | "]" => *depth -= 1,
            _ => {}
        }
        if *depth < 0 {
            return Err(ScriptError::new(number, "a bracket is closed twice"));
        }
        out.push(Spanned {
            token: Token::Symbol(symbol),
            line: number,
        });
        i += symbol.len();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(source: &str) -> Vec<Token> {
        scan(source)
            .expect("scans")
            .into_iter()
            .map(|s| s.token)
            .collect()
    }

    #[test]
    fn indentation_becomes_explicit_blocks() {
        let t = tokens("if x:\n    y = 1\nz = 2\n");
        let indents = t.iter().filter(|t| **t == Token::Indent).count();
        let dedents = t.iter().filter(|t| **t == Token::Dedent).count();
        assert_eq!(indents, 1);
        assert_eq!(dedents, 1);
    }

    /// Blank lines and comments inside a block do not close it.
    #[test]
    fn a_blank_line_does_not_end_a_block() {
        let t = tokens("if x:\n    a = 1\n\n    # still inside\n    b = 2\nc = 3\n");
        assert_eq!(t.iter().filter(|t| **t == Token::Dedent).count(), 1);
    }

    /// Every block left open at the end of the file is closed.
    #[test]
    fn the_end_of_the_file_closes_every_block() {
        let t = tokens("if a:\n    if b:\n        c = 1\n");
        assert_eq!(t.iter().filter(|t| **t == Token::Dedent).count(), 2);
        assert_eq!(t.last(), Some(&Token::End));
    }

    /// Inside brackets a line break is whitespace, so a long call can wrap.
    #[test]
    fn a_line_can_be_continued_inside_brackets() {
        let t = tokens("y = max(1,\n        2)\n");
        assert_eq!(t.iter().filter(|t| **t == Token::Newline).count(), 1);
    }

    #[test]
    fn the_longest_symbol_wins() {
        assert_eq!(
            tokens("a <= b"),
            vec![
                Token::Name("a".into()),
                Token::Symbol("<="),
                Token::Name("b".into()),
                Token::Newline,
                Token::End
            ]
        );
    }

    #[test]
    fn mixed_indentation_is_refused() {
        let error = scan("if x:\n \t a = 1\n").expect_err("mixed indent");
        assert!(error.message.contains("tabs and spaces"), "{error}");
    }

    #[test]
    fn a_stray_character_says_where_it_is() {
        let error = scan("a = 1\nb = 2 $ 3\n").expect_err("stray");
        assert_eq!(error.line, 2);
    }
}
