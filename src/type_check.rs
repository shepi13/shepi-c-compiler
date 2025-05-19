use crate::parser::{
    self, BlockItem, Declaration, Expression, ForInit, FunctionDeclaration, Loop, Program,
    Statement, VariableDeclaration,
};
use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, PartialEq, Clone)]
pub enum CType {
    INT,
    FUNCTION(usize),
}

#[derive(Debug, Clone)]
pub struct Symbol {
    pub name: String,
    pub ctype: CType,
    pub external: bool,
    pub defined: bool,
}

pub type SymbolMap = HashMap<String, Symbol>;
pub struct SymbolTable {
    // Stacks, only used for type checking
    symbols: Vec<SymbolMap>,
    labels: Vec<HashSet<String>>,
    gotos: Vec<HashSet<String>>,
    loops: Vec<String>,
    current_function: String,

    // Global symbols
    extern_symbols: SymbolMap,
}

impl SymbolTable {
    fn new() -> Self {
        Self {
            symbols: vec![HashMap::new()],
            labels: vec![HashSet::new()],
            gotos: vec![HashSet::new()],
            loops: Vec::new(),
            extern_symbols: HashMap::new(),
            current_function: String::new(),
        }
    }
    fn current_loop(&self) -> &str {
        if self.loops.len() == 0 {
            panic!("Used Break/Continue outside Loop!");
        }
        &self.loops[self.loops.len() - 1]
    }
    fn enter_scope(&mut self) {
        self.symbols.push(HashMap::new());
    }
    fn enter_function(&mut self, name: String) {
        self.labels.push(HashSet::new());
        self.gotos.push(HashSet::new());
        self.current_function = name;
    }
    fn enter_loop(&mut self, label: String) {
        self.loops.push(label)
    }
    fn leave_loop(&mut self) {
        self.loops.pop();
    }
    fn leave_scope(&mut self) {
        self.symbols.pop();
    }
    fn leave_function(&mut self) {
        let labels = self.labels.pop().expect("Scope missing from symbol table");
        let gotos = self.gotos.pop().expect("Scope missing from symbol table");

        let mut set_diff = gotos.difference(&labels);
        match set_diff.next() {
            Some(label) => panic!("Tried to goto '{}', but label doesn't exist", label),
            _ => (),
        };
    }
    fn declare_variable(&mut self, name: String) -> String {
        let stacklen = self.symbols.len() - 1;
        let table = &mut self.symbols[stacklen];
        if table.contains_key(&name) {
            panic!("Duplicate variable name in current scope: {}", name);
        }
        let unique_name = variable_name(&name);
        table.insert(
            name,
            Symbol {
                name: unique_name.clone(),
                ctype: CType::INT,
                external: false,
                defined: false,
            },
        );
        unique_name
    }
    fn resolve_variable(&self, name: &str) -> String {
        for table in self.symbols.iter().rev() {
            if table.contains_key(name) {
                assert!(table[name].ctype == CType::INT, "Expected a variable!");
                return table[name].name.clone();
            }
        }
        panic!("Undeclared variable: {}", name);
    }
    fn declare_function(&mut self, name: String, params: &Vec<String>, defined: bool) {
        let stacklen = self.symbols.len() - 1;
        let table = &mut self.symbols[stacklen];
        assert!(
            stacklen == 0 || !defined,
            "Nested functin definitions not allowed"
        );
        match self.extern_symbols.get(&name) {
            Some(symbol) => {
                assert!(
                    symbol.external,
                    "Duplicate function definition in current scope: {}",
                    name
                );
                assert!(
                    symbol.ctype == CType::FUNCTION(params.len()),
                    "Incompatible function definitions, arguments of previously declared function are different types"
                );
                assert!(!symbol.defined || !defined, "Illegal function redefinition");
            }
            None => {
                self.extern_symbols.insert(
                    name.to_string(),
                    Symbol {
                        name: name.to_string(),
                        ctype: CType::FUNCTION(params.len()),
                        external: true,
                        defined,
                    },
                );
            }
        };
        if table.contains_key(&name) {
            assert!(
                table[&name].ctype == CType::FUNCTION(params.len()),
                "Incompatible Function definitions"
            );
        } else {
            table.insert(
                name.clone(),
                Symbol {
                    name: name,
                    ctype: CType::FUNCTION(params.len()),
                    external: true,
                    defined,
                },
            );
        }
    }
    fn resolve_function(&self, name: String, params: &Vec<Expression>) -> String {
        for table in self.symbols.iter().rev() {
            if table.contains_key(&name) {
                assert!(
                    table[&name].ctype == CType::FUNCTION(params.len()),
                    "Incompatible function call!"
                );
                return table[&name].name.clone();
            }
        }
        panic!("Undeclared function: {}", name);
    }
    fn resolve_label(&mut self, target: String) -> String {
        assert!(
            !self.current_function.is_empty(),
            "Labels must be in functions!"
        );
        for table in &self.labels {
            if table.contains(&target) {
                panic!("Duplicate label: {}", target);
            }
        }
        let mangled_label = format!("{}_{}_{}", target, self.current_function, target);
        let stacklen = &self.labels.len() - 1;
        self.labels[stacklen].insert(target);
        mangled_label
    }
    fn resolve_goto(&mut self, target: String) -> String {
        assert!(
            !self.current_function.is_empty(),
            "Can only use goto inside function!"
        );
        let mangled_label = format!("{}_{}_{}", target, self.current_function, target);
        let stacklen = &self.gotos.len() - 1;
        self.gotos[stacklen].insert(target);
        mangled_label
    }
}

fn variable_name(name: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("{}.{}", name, COUNTER.fetch_add(1, Ordering::Relaxed))
}
#[derive(Debug)]
pub struct ResolvedProgram<'a> {
    pub program: Program<'a>,
    pub globals: SymbolMap,
}
pub fn type_check(program: Program) -> ResolvedProgram {
    let mut symbols = SymbolTable::new();
    ResolvedProgram {
        program: program
            .into_iter()
            .map(|f| resolve_function(f, &mut symbols))
            .collect(),
        globals: symbols.extern_symbols,
    }
}

pub fn resolve_function<'a>(
    mut function: FunctionDeclaration<'a>,
    symbols: &mut SymbolTable,
) -> FunctionDeclaration<'a> {
    symbols.declare_function(
        function.name.to_string(),
        &function.params,
        function.body.is_some(),
    );
    symbols.enter_scope();
    function.params = function
        .params
        .into_iter()
        .map(|param| symbols.declare_variable(param))
        .collect();
    if let Some(body) = function.body {
        symbols.enter_function(function.name.to_string());
        function.body = Some(
            body.into_iter()
                .map(|item| resolve_block_item(item, symbols))
                .collect(),
        );
        symbols.leave_function();
    }
    symbols.leave_scope();
    function
}
fn resolve_block_item<'a>(block_item: BlockItem<'a>, symbols: &mut SymbolTable) -> BlockItem<'a> {
    match block_item {
        BlockItem::STATEMENT(statement) => {
            BlockItem::STATEMENT(resolve_statement(statement, symbols))
        }
        BlockItem::DECLARATION(decl) => BlockItem::DECLARATION(resolve_declaration(decl, symbols)),
    }
}
pub fn resolve_statement<'a>(statement: Statement<'a>, symbols: &mut SymbolTable) -> Statement<'a> {
    use parser::Statement::*;
    match statement {
        RETURN(expr) => RETURN(resolve_expression(expr, symbols)),
        EXPRESSION(expr) => EXPRESSION(resolve_expression(expr, symbols)),
        WHILE(loop_data) => WHILE(resolve_loop(loop_data, symbols, true)),
        DOWHILE(loop_data) => DOWHILE(resolve_loop(loop_data, symbols, true)),
        IF(cond, if_true, if_false) => {
            symbols.enter_scope();
            let result = IF(
                resolve_expression(cond, symbols),
                resolve_statement(*if_true, symbols).into(),
                match *if_false {
                    Some(if_false) => Some(resolve_statement(if_false, symbols)),
                    None => None,
                }
                .into(),
            );
            symbols.leave_scope();
            result
        }
        BREAK(_) => BREAK(symbols.current_loop().to_string()),
        CONTINUE(_) => CONTINUE(symbols.current_loop().to_string()),
        COMPOUND(statements) => {
            symbols.enter_scope();
            let result = COMPOUND(
                statements
                    .into_iter()
                    .map(|item| resolve_block_item(item, symbols))
                    .collect(),
            );
            symbols.leave_scope();
            result
        }
        GOTO(label) => GOTO(symbols.resolve_goto(label)),
        LABEL(label, statement) => LABEL(
            symbols.resolve_label(label),
            resolve_statement(*statement, symbols).into(),
        ),
        FOR(init, loop_data, post) => {
            symbols.enter_scope();
            let init = match init {
                ForInit::INITDECL(decl) => ForInit::INITDECL(resolve_variable(decl, symbols)),
                ForInit::INITEXP(expr) => {
                    ForInit::INITEXP(resolve_optional_expression(expr, symbols))
                }
            };
            let post = resolve_optional_expression(post, symbols);
            let loop_data = resolve_loop(loop_data, symbols, false);
            symbols.leave_scope();
            FOR(init, loop_data, post)
        }
        NULL => NULL,
    }
}

fn resolve_declaration<'a>(decl: Declaration<'a>, symbols: &mut SymbolTable) -> Declaration<'a> {
    match decl {
        Declaration::VARIABLE(var_decl) => {
            Declaration::VARIABLE(resolve_variable(var_decl, symbols))
        }
        Declaration::FUNCTION(func_decl) => {
            Declaration::FUNCTION(resolve_function(func_decl, symbols))
        }
    }
}

fn resolve_variable(
    mut decl: VariableDeclaration,
    symbols: &mut SymbolTable,
) -> VariableDeclaration {
    decl.name = symbols.declare_variable(decl.name);
    decl.value = resolve_optional_expression(decl.value, symbols);
    decl
}

fn resolve_expression(expr: Expression, symbols: &mut SymbolTable) -> Expression {
    use parser::Expression::*;
    match expr {
        UNARY(operator, expr) => {
            if matches!(operator, parser::UnaryOperator::INCREMENT(_)) {
                assert!(matches!(*expr, Expression::VAR(_)), "Invalid lvalue!");
            }
            UNARY(operator, resolve_expression(*expr, symbols).into())
        }
        BINARY(mut binexpr) => {
            binexpr.left = resolve_expression(binexpr.left, symbols);
            binexpr.right = resolve_expression(binexpr.right, symbols);
            BINARY(binexpr)
        }
        CONDITION(mut cond) => {
            cond.condition = resolve_expression(cond.condition, symbols);
            cond.if_true = resolve_expression(cond.if_true, symbols);
            cond.if_false = resolve_expression(cond.if_false, symbols);
            CONDITION(cond)
        }
        ASSIGNMENT(mut assign) => {
            assert!(matches!(assign.left, Expression::VAR(_)), "Invalid lvalue!");
            assign.left = resolve_expression(assign.left, symbols);
            assign.right = resolve_expression(assign.right, symbols);
            ASSIGNMENT(assign)
        }
        VAR(name) => VAR(symbols.resolve_variable(&name)),
        LITEXP(_) => expr,
        FUNCTION(name, args) => {
            let name = symbols.resolve_function(name, &args);
            let args = args
                .into_iter()
                .map(|arg| resolve_expression(arg, symbols))
                .collect();
            FUNCTION(name, args)
        }
    }
}

fn resolve_optional_expression(
    expr: Option<Expression>,
    symbols: &mut SymbolTable,
) -> Option<Expression> {
    match expr {
        Some(expr) => Some(resolve_expression(expr, symbols)),
        None => None,
    }
}

pub fn resolve_loop<'a>(
    mut loop_data: Loop<'a>,
    symbols: &mut SymbolTable,
    scoped: bool,
) -> Loop<'a> {
    symbols.enter_loop(loop_data.label.clone());
    if scoped {
        symbols.enter_scope();
    }
    loop_data.condition = resolve_expression(loop_data.condition, symbols);
    loop_data.body = resolve_statement(*loop_data.body, symbols).into();
    symbols.leave_loop();
    if scoped {
        symbols.leave_scope();
    }
    loop_data
}
