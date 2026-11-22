//! Tokens to a tree.
//!
//! A recursive-descent parser with a precedence-climbing expression rule,
//! which is the shape every hand-written parser for an infix language ends up
//! being. There is no table and no generator: the grammar is small enough to
//! read in one sitting, and keeping it that way is a design constraint rather
//! than an accident — see [`crate`] for what is deliberately not in it.

use crate::lex::{Spanned, Token};
use crate::ScriptError;

/// Something that produces a value.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Number(f32),
    Bool(bool),
    Nothing,
    /// A bare name: a local, or one of the globals the host supplies.
    Name(String),
    /// `a.b`
    Field(Box<Expr>, String),
    /// `f(a, b)`
    Call(String, Vec<Expr>),
    /// `a.length()` — a call on the result of an expression.
    Method(Box<Expr>, String, Vec<Expr>),
    Unary(&'static str, Box<Expr>),
    Binary(&'static str, Box<Expr>, Box<Expr>),
}

/// Where an assignment puts its value.
///
/// A path rather than an expression, because the left of an `=` is not
/// evaluated — it is resolved. `a.b.c = 1` names a place; evaluating it would
/// produce the value that is there now, which is exactly what is about to be
/// thrown away.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    pub root: String,
    pub path: Vec<String>,
}

/// Something that happens.
#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Assign(Target, Expr),
    /// `+=` and friends, kept as their own node so the evaluator reads the
    /// target once rather than twice.
    Update(Target, &'static str, Expr),
    If(Vec<(Expr, Vec<Stmt>)>, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
    /// An expression evaluated for its effect, which in this language means a
    /// call. Kept so that a line like `print(x)` parses rather than erroring.
    Eval(Expr),
    Pass,
}

/// A statement, and the line it started on.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub stmt: Stmt,
    pub line: u32,
}

/// How deeply an expression may nest — brackets, call arguments, chains of
/// unary signs, `**` exponents, `not`.
///
/// The parser is recursive descent, so every level is a stack frame, and with
/// no limit `x = ((((…1…))))` ten thousand deep overflowed the stack. A stack
/// overflow is not an error a caller can catch: it aborted the process, and
/// since a script travels inside a `.fusion` file, opening a corrupted or
/// hostile project took the editor down with it. CPython refuses brackets
/// nested past 200; a script nobody wrote by accident stays far under 100.
pub const MAX_NESTING: u32 = 100;

struct Parser<'a> {
    tokens: &'a [Spanned],
    at: usize,
    /// Current expression nesting, against [`MAX_NESTING`].
    depth: u32,
}

/// Parse a whole program.
pub fn parse(tokens: &[Spanned]) -> Result<Vec<Line>, ScriptError> {
    let mut parser = Parser {
        tokens,
        at: 0,
        depth: 0,
    };
    let body = parser.block_until_end()?;
    Ok(body)
}

impl<'a> Parser<'a> {
    /// Recurse one level deeper, or refuse with an ordinary script error.
    fn nested<T>(
        &mut self,
        inner: impl FnOnce(&mut Self) -> Result<T, ScriptError>,
    ) -> Result<T, ScriptError> {
        if self.depth >= MAX_NESTING {
            return Err(ScriptError::new(
                self.line(),
                format!("this expression nests more than {MAX_NESTING} deep"),
            ));
        }
        self.depth += 1;
        let result = inner(self);
        self.depth -= 1;
        result
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.at.min(self.tokens.len() - 1)].token
    }

    fn line(&self) -> u32 {
        self.tokens[self.at.min(self.tokens.len() - 1)].line
    }

    fn advance(&mut self) -> Token {
        let token = self.peek().clone();
        if self.at < self.tokens.len() - 1 {
            self.at += 1;
        }
        token
    }

    fn eat_symbol(&mut self, symbol: &str) -> bool {
        if matches!(self.peek(), Token::Symbol(s) if *s == symbol) {
            self.advance();
            return true;
        }
        false
    }

    fn expect_symbol(&mut self, symbol: &str) -> Result<(), ScriptError> {
        if self.eat_symbol(symbol) {
            return Ok(());
        }
        Err(ScriptError::new(
            self.line(),
            format!("expected {symbol:?} here, found {}", describe(self.peek())),
        ))
    }

    fn eat_keyword(&mut self, word: &str) -> bool {
        if matches!(self.peek(), Token::Name(n) if n == word) {
            self.advance();
            return true;
        }
        false
    }

    fn at_keyword(&self, word: &str) -> bool {
        matches!(self.peek(), Token::Name(n) if n == word)
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }
    }

    fn block_until_end(&mut self) -> Result<Vec<Line>, ScriptError> {
        let mut body = Vec::new();
        loop {
            self.skip_newlines();
            if matches!(self.peek(), Token::End) {
                return Ok(body);
            }
            if matches!(self.peek(), Token::Dedent) {
                // A stray dedent at the top level means the indentation logic
                // and the parser disagree, which is a bug in one of them
                // rather than in the script.
                self.advance();
                continue;
            }
            body.push(self.statement()?);
        }
    }

    /// The body of an `if` or a `while`: either an indented block, or a single
    /// statement on the same line after the colon.
    fn body(&mut self) -> Result<Vec<Line>, ScriptError> {
        if matches!(self.peek(), Token::Newline) {
            self.skip_newlines();
            if !matches!(self.peek(), Token::Indent) {
                return Err(ScriptError::new(
                    self.line(),
                    "this block is empty — indent the lines that belong to it",
                ));
            }
            self.advance();
            let mut body = Vec::new();
            loop {
                self.skip_newlines();
                match self.peek() {
                    Token::Dedent => {
                        self.advance();
                        return Ok(body);
                    }
                    Token::End => return Ok(body),
                    _ => body.push(self.statement()?),
                }
            }
        }
        Ok(vec![self.statement()?])
    }

    fn statement(&mut self) -> Result<Line, ScriptError> {
        let line = self.line();
        let stmt = if self.at_keyword("if") {
            self.advance();
            self.if_statement()?
        } else if self.at_keyword("while") {
            self.advance();
            let condition = self.expression()?;
            self.expect_symbol(":")?;
            let body = self.body()?;
            Stmt::While(condition, unwrap(body))
        } else if self.eat_keyword("pass") {
            Stmt::Pass
        } else {
            self.simple()?
        };
        // A simple statement ends at the newline; a compound one has already
        // consumed its block.
        if matches!(self.peek(), Token::Newline) {
            self.advance();
        }
        Ok(Line { stmt, line })
    }

    fn if_statement(&mut self) -> Result<Stmt, ScriptError> {
        let mut arms = Vec::new();
        let condition = self.expression()?;
        self.expect_symbol(":")?;
        arms.push((condition, unwrap(self.body()?)));

        loop {
            self.skip_newlines();
            if self.at_keyword("elif") {
                self.advance();
                let condition = self.expression()?;
                self.expect_symbol(":")?;
                arms.push((condition, unwrap(self.body()?)));
                continue;
            }
            if self.at_keyword("else") {
                self.advance();
                self.expect_symbol(":")?;
                let otherwise = unwrap(self.body()?);
                return Ok(Stmt::If(arms, otherwise));
            }
            return Ok(Stmt::If(arms, Vec::new()));
        }
    }

    /// An assignment, an update, or a bare expression.
    ///
    /// Which of the three is not knowable until the `=` is reached, so this
    /// parses an expression first and then decides — the standard trick, and
    /// the reason `a.b = 1` and `a.b` share a code path up to that point.
    fn simple(&mut self) -> Result<Stmt, ScriptError> {
        let line = self.line();
        let left = self.expression()?;

        for (symbol, operator) in [("+=", "+"), ("-=", "-"), ("*=", "*"), ("/=", "/")] {
            if self.eat_symbol(symbol) {
                let value = self.expression()?;
                let target = as_target(&left, line)?;
                return Ok(Stmt::Update(target, operator, value));
            }
        }
        if self.eat_symbol("=") {
            let value = self.expression()?;
            let target = as_target(&left, line)?;
            return Ok(Stmt::Assign(target, value));
        }
        Ok(Stmt::Eval(left))
    }

    // --- expressions ----------------------------------------------------

    fn expression(&mut self) -> Result<Expr, ScriptError> {
        self.or()
    }

    fn or(&mut self) -> Result<Expr, ScriptError> {
        let mut left = self.and()?;
        while self.eat_keyword("or") {
            let right = self.and()?;
            left = Expr::Binary("or", Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Expr, ScriptError> {
        let mut left = self.not()?;
        while self.eat_keyword("and") {
            let right = self.not()?;
            left = Expr::Binary("and", Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not(&mut self) -> Result<Expr, ScriptError> {
        if self.eat_keyword("not") {
            let inner = self.nested(Self::not)?;
            return Ok(Expr::Unary("not", Box::new(inner)));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Result<Expr, ScriptError> {
        let left = self.sum()?;
        for symbol in ["==", "!=", "<=", ">=", "<", ">"] {
            if self.eat_symbol(symbol) {
                let right = self.sum()?;
                let symbol = ["==", "!=", "<=", ">=", "<", ">"]
                    .iter()
                    .find(|s| **s == symbol)
                    .expect("from the same list");
                return Ok(Expr::Binary(symbol, Box::new(left), Box::new(right)));
            }
        }
        Ok(left)
    }

    fn sum(&mut self) -> Result<Expr, ScriptError> {
        let mut left = self.product()?;
        loop {
            let operator = if self.eat_symbol("+") {
                "+"
            } else if self.eat_symbol("-") {
                "-"
            } else {
                return Ok(left);
            };
            let right = self.product()?;
            left = Expr::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn product(&mut self) -> Result<Expr, ScriptError> {
        let mut left = self.unary()?;
        loop {
            let operator = if self.eat_symbol("*") {
                "*"
            } else if self.eat_symbol("/") {
                "/"
            } else if self.eat_symbol("%") {
                "%"
            } else if self.eat_symbol("//") {
                "//"
            } else {
                return Ok(left);
            };
            let right = self.unary()?;
            left = Expr::Binary(operator, Box::new(left), Box::new(right));
        }
    }

    fn unary(&mut self) -> Result<Expr, ScriptError> {
        if self.eat_symbol("-") {
            let inner = self.nested(Self::unary)?;
            return Ok(Expr::Unary("-", Box::new(inner)));
        }
        if self.eat_symbol("+") {
            return self.nested(Self::unary);
        }
        self.power()
    }

    /// `**` binds tighter than unary minus and associates to the right, so
    /// `-2 ** 2` is `-(2 ** 2)` and `2 ** 3 ** 2` is `2 ** 9`. Both are
    /// Python's rules, and both are the kind of thing that is only noticed
    /// when it is wrong.
    fn power(&mut self) -> Result<Expr, ScriptError> {
        let base = self.postfix()?;
        if self.eat_symbol("**") {
            let exponent = self.nested(Self::unary)?;
            return Ok(Expr::Binary("**", Box::new(base), Box::new(exponent)));
        }
        Ok(base)
    }

    fn postfix(&mut self) -> Result<Expr, ScriptError> {
        let mut value = self.primary()?;
        while self.eat_symbol(".") {
            let Token::Name(field) = self.advance() else {
                return Err(ScriptError::new(self.line(), "expected a name after '.'"));
            };
            if self.eat_symbol("(") {
                let arguments = self.arguments()?;
                value = Expr::Method(Box::new(value), field, arguments);
            } else {
                value = Expr::Field(Box::new(value), field);
            }
        }
        Ok(value)
    }

    fn arguments(&mut self) -> Result<Vec<Expr>, ScriptError> {
        let mut arguments = Vec::new();
        if self.eat_symbol(")") {
            return Ok(arguments);
        }
        loop {
            arguments.push(self.nested(Self::expression)?);
            if self.eat_symbol(",") {
                continue;
            }
            self.expect_symbol(")")?;
            return Ok(arguments);
        }
    }

    fn primary(&mut self) -> Result<Expr, ScriptError> {
        let line = self.line();
        match self.advance() {
            Token::Number(value) => Ok(Expr::Number(value)),
            Token::Symbol("(") => {
                let inner = self.nested(Self::expression)?;
                self.expect_symbol(")")?;
                Ok(inner)
            }
            Token::Name(name) => match name.as_str() {
                "True" => Ok(Expr::Bool(true)),
                "False" => Ok(Expr::Bool(false)),
                "None" => Ok(Expr::Nothing),
                _ => {
                    if self.eat_symbol("(") {
                        let arguments = self.arguments()?;
                        Ok(Expr::Call(name, arguments))
                    } else {
                        Ok(Expr::Name(name))
                    }
                }
            },
            other => Err(ScriptError::new(
                line,
                format!("expected a value here, found {}", describe(&other)),
            )),
        }
    }
}

/// Turn an expression that was parsed on the left of an `=` into a place.
fn as_target(expr: &Expr, line: u32) -> Result<Target, ScriptError> {
    let mut path = Vec::new();
    let mut current = expr;
    loop {
        match current {
            Expr::Name(name) => {
                path.reverse();
                return Ok(Target {
                    root: name.clone(),
                    path,
                });
            }
            Expr::Field(inner, field) => {
                path.push(field.clone());
                current = inner;
            }
            _ => {
                return Err(ScriptError::new(
                    line,
                    "the left of an '=' has to be a name or a field, like `self.position.y`",
                ))
            }
        }
    }
}

fn unwrap(lines: Vec<Line>) -> Vec<Stmt> {
    lines.into_iter().map(|l| l.stmt).collect()
}

fn describe(token: &Token) -> String {
    match token {
        Token::Name(name) => format!("{name:?}"),
        Token::Number(value) => format!("{value}"),
        Token::Symbol(symbol) => format!("{symbol:?}"),
        Token::Newline => "the end of the line".into(),
        Token::Indent => "an indent".into(),
        Token::Dedent => "the end of a block".into(),
        Token::End => "the end of the script".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::scan;

    fn tree(source: &str) -> Vec<Line> {
        parse(&scan(source).expect("scans")).expect("parses")
    }

    #[test]
    fn an_assignment_to_a_field_is_a_path() {
        let lines = tree("self.position.y = 3\n");
        let Stmt::Assign(target, _) = &lines[0].stmt else {
            panic!("not an assignment: {:?}", lines[0]);
        };
        assert_eq!(target.root, "self");
        assert_eq!(target.path, vec!["position", "y"]);
    }

    /// Precedence, in the one place it is most often got wrong.
    #[test]
    fn multiplication_binds_tighter_than_addition() {
        let lines = tree("x = 1 + 2 * 3\n");
        let Stmt::Assign(_, Expr::Binary("+", left, right)) = &lines[0].stmt else {
            panic!("wrong shape: {:?}", lines[0]);
        };
        assert_eq!(**left, Expr::Number(1.0));
        assert!(matches!(**right, Expr::Binary("*", _, _)));
    }

    /// `-2 ** 2` is `-(2 ** 2)`, which is Python's rule and not everyone's.
    #[test]
    fn power_binds_tighter_than_unary_minus() {
        let lines = tree("x = -2 ** 2\n");
        let Stmt::Assign(_, Expr::Unary("-", inner)) = &lines[0].stmt else {
            panic!("wrong shape: {:?}", lines[0]);
        };
        assert!(matches!(**inner, Expr::Binary("**", _, _)));
    }

    #[test]
    fn if_elif_else_all_land_in_one_node() {
        let lines = tree("if a:\n    x = 1\nelif b:\n    x = 2\nelse:\n    x = 3\n");
        let Stmt::If(arms, otherwise) = &lines[0].stmt else {
            panic!("not an if: {:?}", lines[0]);
        };
        assert_eq!(arms.len(), 2);
        assert_eq!(otherwise.len(), 1);
    }

    #[test]
    fn a_method_call_is_not_a_field() {
        let lines = tree("x = self.velocity.length()\n");
        let Stmt::Assign(_, Expr::Method(_, name, args)) = &lines[0].stmt else {
            panic!("wrong shape: {:?}", lines[0]);
        };
        assert_eq!(name, "length");
        assert!(args.is_empty());
    }

    /// The error points at the line the mistake is on, not at the end of file.
    #[test]
    fn a_syntax_error_names_its_line() {
        let tokens = scan("a = 1\nb = \nc = 3\n").expect("scans");
        let error = parse(&tokens).expect_err("should not parse");
        assert_eq!(error.line, 2, "{error}");
    }

    #[test]
    fn assigning_to_something_that_is_not_a_place_is_refused() {
        let tokens = scan("1 + 2 = 3\n").expect("scans");
        let error = parse(&tokens).expect_err("should not parse");
        assert!(error.message.contains("left of an '='"), "{error}");
    }
}
