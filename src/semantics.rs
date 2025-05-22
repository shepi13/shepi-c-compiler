use crate::parser::{
    self, BlockItem, Declaration, ForInit, FunctionDeclaration, Loop, Program, Statement,
    StorageClass, SwitchStatement, TypedExpression, VariableDeclaration,
};
use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicUsize, Ordering},
};

#[derive(Debug, Clone)]
pub struct Identifier {
    pub unique_name: String,
    pub external: bool,
}

pub struct SwitchData {
    pub label: String,
    pub cases: Vec<(String, TypedExpression)>,
    pub default: Option<String>,
}

pub struct SymbolTable {
    // Stacks (tracks visibility)
    identifiers: Vec<HashMap<String, Identifier>>,

    labels: Vec<HashSet<String>>,
    gotos: Vec<HashSet<String>>,

    loops: Vec<String>,
    switches: Vec<SwitchData>,

    break_scopes: Vec<String>,
    current_function: Option<String>,
}

impl SymbolTable {
    fn new() -> Self {
        Self {
            identifiers: vec![HashMap::new()],
            labels: Vec::new(),
            gotos: Vec::new(),
            loops: Vec::new(),
            switches: Vec::new(),
            break_scopes: Vec::new(),

            current_function: None,
        }
    }
    fn resolve_continue(&self) -> &str {
        if self.loops.len() == 0 {
            panic!("Used continue outside loop!");
        }
        &self.loops[self.loops.len() - 1]
    }
    fn resolve_break(&self) -> &str {
        if self.break_scopes.len() == 0 {
            panic!("Used break outside loop or switch!");
        }
        &self.break_scopes[self.break_scopes.len() - 1]
    }
    fn resolve_case(&mut self, matcher: TypedExpression) -> String {
        let stack_len = self.switches.len() - 1;
        assert!(self.switches.len() > 0, "Used case outside of switch!");
        let case_label = case_name(&self.switches[stack_len].label);
        self.switches[stack_len]
            .cases
            .push((case_label.clone(), matcher));
        case_label
    }
    fn resolve_default(&mut self) -> String {
        if self.switches.len() == 0 {
            panic!("Used default outside of switch!");
        }
        let stack_len = self.switches.len() - 1;
        assert!(
            matches!(&self.switches[stack_len].default, None),
            "duplicate default!"
        );
        let default_label = format!("{}.default", &self.switches[stack_len].label);
        self.switches[stack_len].default = Some(default_label.clone());
        default_label
    }
    fn enter_scope(&mut self) {
        self.identifiers.push(HashMap::new());
    }
    fn leave_scope(&mut self) {
        self.identifiers.pop();
    }
    fn enter_loop(&mut self, label: String) {
        self.loops.push(label.clone());
        self.break_scopes.push(label);
    }
    fn leave_loop(&mut self) {
        self.loops.pop();
        self.break_scopes.pop();
    }
    fn enter_switch(&mut self, switch: &SwitchStatement) {
        self.break_scopes.push(switch.label.to_string());
        self.switches.push(SwitchData {
            label: switch.label.to_string(),
            cases: Vec::new(),
            default: None,
        });
    }
    fn leave_switch(&mut self) -> SwitchData {
        self.break_scopes.pop();
        self.switches
            .pop()
            .expect("Scope missing from symbol table")
    }
    fn enter_function(&mut self, name: String) {
        self.labels.push(HashSet::new());
        self.gotos.push(HashSet::new());
        self.current_function = Some(name);
    }
    fn leave_function(&mut self) {
        let labels = self.labels.pop().expect("Scope missing from symbol table");
        let gotos = self.gotos.pop().expect("Scope missing from symbol table");

        let mut set_diff = gotos.difference(&labels);
        match set_diff.next() {
            Some(label) => panic!("Tried to goto '{}', but label doesn't exist", label),
            _ => (),
        };
        self.current_function = None;
    }
    fn _lookup_identifier(&self, name: &str) -> Option<&Identifier> {
        for table in self.identifiers.iter().rev() {
            if let Some(identifier) = table.get(name) {
                return Some(identifier);
            }
        }
        None
    }
    fn declare_global_variable(&mut self, name: String) {
        let stack_len = self.identifiers.len() - 1;
        self.identifiers[stack_len].insert(
            name.clone(),
            Identifier {
                unique_name: name.clone(),
                external: true,
            },
        );
    }
    fn declare_parameter(&mut self, name: String) -> String {
        let decl = self.declare_local_variable(VariableDeclaration {
            name: name.clone(),
            value: None,
            ctype: parser::CType::Int,
            storage: None,
        });
        decl.name
    }
    fn declare_local_variable(&mut self, mut decl: VariableDeclaration) -> VariableDeclaration {
        let name = &decl.name;
        let stack_len = self.identifiers.len() - 1;
        if let Some(prev_declaration) = self.identifiers[stack_len].get(name) {
            println!("{:?} and {:?}", prev_declaration.external, decl.storage);
            assert!(
                prev_declaration.external && decl.storage == Some(StorageClass::Extern),
                "Duplicate variable name in current scope: {}",
                name
            )
        }
        if decl.storage == Some(StorageClass::Extern) {
            self.declare_global_variable(decl.name.clone());
            decl
        } else {
            let unique_name = variable_name(&name);
            self.identifiers[stack_len].insert(
                decl.name,
                Identifier {
                    unique_name: unique_name.clone(),
                    external: false,
                },
            );
            decl.name = unique_name;
            decl
        }
    }
    fn declare_function(&mut self, function: &FunctionDeclaration) {
        let defined = function.body.is_some();
        let name = &function.name;
        if self.current_function.is_some() {
            assert!(!defined, "Nested function definitions not allowed");
            assert!(
                function.storage != Some(StorageClass::Static),
                "static only allowed at top level scope"
            );
        }
        let stack_len = self.identifiers.len() - 1;
        if let Some(ident) = self.identifiers[stack_len].get(name) {
            assert!(ident.external, "Duplicate function declaration: {}", name);
        }
        self.identifiers[stack_len].insert(
            name.clone(),
            Identifier {
                unique_name: name.clone(),
                external: true,
            },
        );
    }
    fn resolve_identifier(&self, name: &str) -> String {
        let ident = self
            ._lookup_identifier(name)
            .expect(&format!("Undeclared variable {}", name));
        ident.unique_name.to_string()
    }
    fn resolve_label(&mut self, target: String) -> String {
        let function = self
            .current_function
            .as_ref()
            .expect("Labels must be in functions");
        for table in &self.labels {
            if table.contains(&target) {
                panic!("Duplicate label: {}", target);
            }
        }
        let stack_len = self.labels.len() - 1;
        let mangled_label = format!("{}_{}_{}", target, function, target);
        self.labels[stack_len].insert(target);
        mangled_label
    }
    fn resolve_goto(&mut self, target: String) -> String {
        let stack_len = self.gotos.len() - 1;
        let function = &self
            .current_function
            .as_ref()
            .expect("Goto must be in function!");
        let mangled_label = format!("{}_{}_{}", target, function, target);
        self.gotos[stack_len].insert(target);
        mangled_label
    }
}

fn case_name(name: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("{}.case_{}", name, COUNTER.fetch_add(1, Ordering::Relaxed))
}
fn variable_name(name: &str) -> String {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    format!("{}.{}", name, COUNTER.fetch_add(1, Ordering::Relaxed))
}

pub fn resolve_program(program: Program) -> Program {
    let symbols = &mut SymbolTable::new();
    program
        .into_iter()
        .map(|decl| resolve_declaration(decl, symbols, true))
        .collect()
}

fn resolve_function(
    mut function: FunctionDeclaration,
    symbols: &mut SymbolTable,
) -> FunctionDeclaration {
    symbols.declare_function(&function);
    symbols.enter_scope();
    function.params = function
        .params
        .into_iter()
        .map(|param| symbols.declare_parameter(param))
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
fn resolve_block_item<'a>(block_item: BlockItem, symbols: &mut SymbolTable) -> BlockItem {
    match block_item {
        BlockItem::StatementItem(statement) => {
            BlockItem::StatementItem(resolve_statement(statement, symbols))
        }
        BlockItem::DeclareItem(decl) => {
            BlockItem::DeclareItem(resolve_declaration(decl, symbols, false))
        }
    }
}
fn resolve_statement<'a>(statement: Statement, symbols: &mut SymbolTable) -> Statement {
    use parser::Statement::*;
    match statement {
        Return(expr) => Return(resolve_expression(expr, symbols)),
        ExprStmt(expr) => ExprStmt(resolve_expression(expr, symbols)),
        While(loop_data) => While(resolve_loop(loop_data, symbols, true)),
        DoWhile(loop_data) => DoWhile(resolve_loop(loop_data, symbols, true)),
        If(cond, if_true, if_false) => {
            symbols.enter_scope();
            let result = If(
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
        Break(_) => Break(symbols.resolve_break().to_string()),
        Continue(_) => Continue(symbols.resolve_continue().to_string()),
        Compound(statements) => {
            symbols.enter_scope();
            let result = Compound(
                statements
                    .into_iter()
                    .map(|item| resolve_block_item(item, symbols))
                    .collect(),
            );
            symbols.leave_scope();
            result
        }
        Goto(label) => Goto(symbols.resolve_goto(label)),
        Label(label, statement) => Label(
            symbols.resolve_label(label),
            resolve_statement(*statement, symbols).into(),
        ),
        For(init, loop_data, post) => {
            symbols.enter_scope();
            let init = match init {
                ForInit::Decl(decl) => {
                    assert!(
                        decl.storage.is_none(),
                        "For loop initializer cannot be static or extern"
                    );
                    ForInit::Decl(resolve_variable_declaration(decl, symbols, false))
                }
                ForInit::Expr(expr) => ForInit::Expr(resolve_optional_expression(expr, symbols)),
            };
            let post = resolve_optional_expression(post, symbols);
            let loop_data = resolve_loop(loop_data, symbols, false);
            symbols.leave_scope();
            For(init, loop_data, post)
        }
        Case(matcher, statement) => Label(
            symbols.resolve_case(matcher),
            resolve_statement(*statement, symbols).into(),
        ),
        Default(statement) => Label(
            symbols.resolve_default(),
            resolve_statement(*statement, symbols).into(),
        ),
        Switch(mut switch) => {
            symbols.enter_switch(&switch);
            switch.condition = resolve_expression(switch.condition, symbols);
            switch.statement = resolve_statement(*switch.statement, symbols).into();
            let switch_symbols = symbols.leave_switch();
            switch.cases = switch_symbols.cases;
            switch.label = switch_symbols.label;
            switch.default = switch_symbols.default;
            Switch(switch)
        }
        Null => Null,
    }
}

fn resolve_declaration(decl: Declaration, symbols: &mut SymbolTable, global: bool) -> Declaration {
    match decl {
        Declaration::Variable(var_decl) => {
            Declaration::Variable(resolve_variable_declaration(var_decl, symbols, global))
        }
        Declaration::Function(func_decl) => {
            Declaration::Function(resolve_function(func_decl, symbols))
        }
    }
}

fn resolve_variable_declaration(
    mut decl: VariableDeclaration,
    symbols: &mut SymbolTable,
    global: bool,
) -> VariableDeclaration {
    if global {
        symbols.declare_global_variable(decl.name.clone());
    } else {
        decl = symbols.declare_local_variable(decl)
    };
    decl.value = resolve_optional_expression(decl.value, symbols);
    decl
}

fn resolve_expression(expr: TypedExpression, symbols: &mut SymbolTable) -> TypedExpression {
    use parser::Expression::*;
    let expr = expr.expr;
    match expr {
        Unary(operator, expr) => {
            if matches!(operator, parser::UnaryOperator::Increment(_)) {
                assert!(matches!(expr.expr, Variable(_)), "Invalid lvalue!");
            }
            Unary(operator, resolve_expression(*expr, symbols).into()).into()
        }
        Binary(mut binexpr) => {
            if binexpr.is_assignment {
                assert!(matches!(binexpr.left.expr, Variable(_)), "Invalid lvalue!");
            }
            binexpr.left = resolve_expression(binexpr.left, symbols);
            binexpr.right = resolve_expression(binexpr.right, symbols);
            Binary(binexpr).into()
        }
        Condition(mut cond) => {
            cond.condition = resolve_expression(cond.condition, symbols);
            cond.if_true = resolve_expression(cond.if_true, symbols);
            cond.if_false = resolve_expression(cond.if_false, symbols);
            Condition(cond).into()
        }
        Assignment(mut assign) => {
            assert!(matches!(assign.left.expr, Variable(_)), "Invalid lvalue!");
            assign.left = resolve_expression(assign.left, symbols);
            assign.right = resolve_expression(assign.right, symbols);
            Assignment(assign).into()
        }
        Variable(name) => Variable(symbols.resolve_identifier(&name)).into(),
        Constant(_) => expr.into(),
        FunctionCall(name, args) => {
            let name = symbols.resolve_identifier(&name);
            let args = args
                .into_iter()
                .map(|arg| resolve_expression(arg, symbols))
                .collect();
            FunctionCall(name, args).into()
        }
        Cast(new_type, expr) => {
            Cast(new_type, resolve_expression(*expr, symbols).into()).into()
        }
    }
}

fn resolve_optional_expression(
    expr: Option<TypedExpression>,
    symbols: &mut SymbolTable,
) -> Option<TypedExpression> {
    match expr {
        Some(expr) => Some(resolve_expression(expr, symbols)),
        None => None,
    }
}

fn resolve_loop(mut loop_data: Loop, symbols: &mut SymbolTable, scoped: bool) -> Loop {
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
