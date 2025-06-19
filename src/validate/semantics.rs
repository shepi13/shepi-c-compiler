use crate::{
    helpers::error::{AddLocation, Error, assert_or_err},
    parse::parse_tree::{
        self, BlockItem, Declaration, ForInit, FunctionDeclaration, Location, Loop, Program,
        Statement, StorageClass, TypedExpression, VariableDeclaration, VariableInitializer,
    },
    validate::ctype::CType,
};
use std::{
    collections::{HashMap, HashSet},
    sync::atomic::{AtomicUsize, Ordering},
};

type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone)]
struct Identifier {
    pub unique_name: String,
    pub external: bool,
}

struct SwitchData {
    pub label: String,
    pub cases: Vec<(String, TypedExpression)>,
    pub default: Option<String>,
}

/// Local symbol table for checking scope of local variables, loops, and labels
struct SymbolTable {
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
    fn resolve_continue(&self) -> Result<&str> {
        assert_or_err(
            !self.loops.is_empty(),
            "Illegal continue",
            "Used continue outside of loop!",
        )?;
        Ok(&self.loops[self.loops.len() - 1])
    }
    fn resolve_break(&self) -> Result<&str> {
        assert_or_err(
            !self.break_scopes.is_empty(),
            "Illegal break",
            "Used break outside loop or switch!",
        )?;
        Ok(&self.break_scopes[self.break_scopes.len() - 1])
    }
    fn resolve_case(&mut self, matcher: TypedExpression) -> Result<String> {
        let stack_len = self.switches.len() - 1;
        assert_or_err(
            !self.switches.is_empty(),
            "Illegal case statement",
            "Used case outside of switch!",
        )?;
        let case_label = case_name(&self.switches[stack_len].label);
        self.switches[stack_len].cases.push((case_label.clone(), matcher));
        Ok(case_label)
    }
    fn resolve_default(&mut self) -> Result<String> {
        assert_or_err(
            !self.switches.is_empty(),
            "Illegal default statement",
            "Used default outside of switch!",
        )?;
        let stack_len = self.switches.len() - 1;
        assert_or_err(
            self.switches[stack_len].default.is_none(),
            "Illegal default statement",
            "duplicate default!",
        )?;
        let default_label = format!("{}.default", &self.switches[stack_len].label);
        self.switches[stack_len].default = Some(default_label.clone());
        Ok(default_label)
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
    fn enter_switch(&mut self, switch_name: String) {
        self.break_scopes.push(switch_name.clone());
        self.switches.push(SwitchData {
            label: switch_name,
            cases: Vec::new(),
            default: None,
        });
    }
    fn leave_switch(&mut self) -> SwitchData {
        self.break_scopes.pop();
        self.switches.pop().expect("Scope missing from symbol table")
    }
    fn enter_function(&mut self, name: String) {
        self.labels.push(HashSet::new());
        self.gotos.push(HashSet::new());
        self.current_function = Some(name);
    }
    fn leave_function(&mut self) {
        let labels = self.labels.pop().expect("Scope missing from symbol table");
        let gotos = self.gotos.pop().expect("Scope missing from symbol table");

        let set_diff = gotos.difference(&labels).next();
        assert!(
            set_diff.is_none(),
            "Tried to goto '{}', but label doesn't exist",
            set_diff.unwrap()
        );
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
    fn declare_parameter(&mut self, name: String) -> Result<String> {
        let decl = self.declare_local_variable(VariableDeclaration {
            name: name.clone(),
            init: None,
            ctype: CType::Int,
            storage: None,
            location: Location { start_loc: (0, 0), end_loc: (0, 0) },
        })?;
        Ok(decl.name)
    }
    fn declare_local_variable(
        &mut self,
        mut decl: VariableDeclaration,
    ) -> Result<VariableDeclaration> {
        let name = &decl.name;
        let stack_len = self.identifiers.len() - 1;
        if let Some(prev_declaration) = self.identifiers[stack_len].get(name) {
            assert_or_err(
                prev_declaration.external && decl.storage == Some(StorageClass::Extern),
                "Illegal declaration",
                "Duplicate variable name in current scope",
            )?;
        }
        if decl.storage == Some(StorageClass::Extern) {
            self.declare_global_variable(decl.name.clone());
            Ok(decl)
        } else {
            let unique_name = variable_name(name);
            self.identifiers[stack_len].insert(
                decl.name,
                Identifier {
                    unique_name: unique_name.clone(),
                    external: false,
                },
            );
            decl.name = unique_name;
            Ok(decl)
        }
    }
    fn declare_function(&mut self, function: &FunctionDeclaration) -> Result<()> {
        let defined = function.body.is_some();
        let name = &function.name;
        if self.current_function.is_some() {
            assert_or_err(
                !defined,
                "Illegal declaration",
                "Nested function definitions not allowed",
            )?;
            assert_or_err(
                function.storage != Some(StorageClass::Static),
                "Illegal declaration",
                "static only allowed at top level scope",
            )?;
        }
        let stack_len = self.identifiers.len() - 1;
        if let Some(ident) = self.identifiers[stack_len].get(name) {
            assert_or_err(ident.external, "Illegal declaration", "Duplicate function declaration")?;
        }
        self.identifiers[stack_len].insert(
            name.clone(),
            Identifier {
                unique_name: name.clone(),
                external: true,
            },
        );
        Ok(())
    }
    fn resolve_identifier(&self, name: &str) -> Result<String> {
        let ident = self._lookup_identifier(name).ok_or_else(|| {
            Error::new("Invalid identifier", &format!("Undeclared variable: {}", name))
        })?;
        Ok(ident.unique_name.to_string())
    }
    fn resolve_label(&mut self, target: String) -> Result<String> {
        let function = self
            .current_function
            .as_ref()
            .ok_or(Error::new("Invalid label", "Labels must be in functions"))?;
        for table in &self.labels {
            if table.contains(&target) {
                return Err(Error::new(
                    "Invalid label",
                    &format!("Duplicate label declaration: {}", target),
                ));
            }
        }
        let stack_len = self.labels.len() - 1;
        let mangled_label = format!("{}_{}_{}", target, function, target);
        self.labels[stack_len].insert(target);
        Ok(mangled_label)
    }
    fn resolve_goto(&mut self, target: String) -> Result<String> {
        let stack_len = self.gotos.len() - 1;
        let function = &self
            .current_function
            .as_ref()
            .ok_or(Error::new("Invalid goto", "Goto must be in function!"))?;
        let mangled_label = format!("{}_{}_{}", target, function, target);
        self.gotos[stack_len].insert(target);
        Ok(mangled_label)
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

pub fn resolve_program(program: Program) -> Result<Program> {
    let symbols = &mut SymbolTable::new();
    let new_program: Result<Vec<Declaration>> =
        program.into_iter().map(|decl| resolve_declaration(decl, symbols, true)).collect();
    new_program
}

fn resolve_function(
    mut function: FunctionDeclaration,
    symbols: &mut SymbolTable,
) -> Result<FunctionDeclaration> {
    symbols.declare_function(&function)?;
    symbols.enter_scope();
    function.params = function
        .params
        .into_iter()
        .map(|param| symbols.declare_parameter(param))
        .collect::<Result<Vec<String>>>()?;
    if let Some(body) = function.body {
        symbols.enter_function(function.name.to_string());
        function.body = Some(
            body.into_iter()
                .map(|item| resolve_block_item(item, symbols))
                .collect::<Result<Vec<BlockItem>>>()?,
        );
        symbols.leave_function();
    }
    symbols.leave_scope();
    Ok(function)
}
fn resolve_block_item(block_item: BlockItem, symbols: &mut SymbolTable) -> Result<BlockItem> {
    match block_item {
        BlockItem::StatementItem(statement, location) => {
            let stmt = resolve_statement(statement, symbols).add_location(location)?;
            Ok(BlockItem::StatementItem(stmt, location))
        }
        BlockItem::DeclareItem(decl) => {
            Ok(BlockItem::DeclareItem(resolve_declaration(decl, symbols, false)?))
        }
    }
}
fn resolve_statement(statement: Statement, symbols: &mut SymbolTable) -> Result<Statement> {
    use parse_tree::Statement::*;
    let new_statement = match statement {
        Return(Some(expr)) => Return(Some(resolve_expression(expr, symbols)?)),
        Return(None) => todo!("Void return types semantics!"),
        ExprStmt(expr) => ExprStmt(resolve_expression(expr, symbols)?),
        While(loop_data) => While(resolve_loop(loop_data, symbols, true)?),
        DoWhile(loop_data) => DoWhile(resolve_loop(loop_data, symbols, true)?),
        If(cond, if_true, if_false) => {
            symbols.enter_scope();
            let result = If(
                resolve_expression(cond, symbols)?,
                resolve_statement(*if_true, symbols)?.into(),
                if_false.map(|stmt| resolve_statement(stmt, symbols)).transpose()?.into(),
            );
            symbols.leave_scope();
            result
        }
        Break(_) => Break(symbols.resolve_break()?.to_string()),
        Continue(_) => Continue(symbols.resolve_continue()?.to_string()),
        Compound(statements) => {
            symbols.enter_scope();
            let new_statements: Result<Vec<BlockItem>> =
                statements.into_iter().map(|item| resolve_block_item(item, symbols)).collect();
            let result = Compound(new_statements?);
            symbols.leave_scope();
            result
        }
        Goto(label) => Goto(symbols.resolve_goto(label)?),
        Label(label, statement) => {
            Label(symbols.resolve_label(label)?, resolve_statement(*statement, symbols)?.into())
        }
        For(init, loop_data, post) => {
            symbols.enter_scope();
            let init = match *init {
                ForInit::Decl(decl) => {
                    assert_or_err(
                        decl.storage.is_none(),
                        "Invalid for-loop initializer",
                        "For loop initializer cannot be static or extern",
                    )?;
                    let mut decl = symbols.declare_local_variable(decl)?;
                    decl.init = resolve_initializer(decl.init, symbols)?;
                    ForInit::Decl(decl)
                }
                ForInit::Expr(expr) => {
                    let expr = match expr {
                        Some(expr) => Some(resolve_expression(expr, symbols)?),
                        None => None,
                    };
                    ForInit::Expr(expr)
                }
            };
            let post = post.map(|expr| resolve_expression(expr, symbols)).transpose()?;
            let loop_data = resolve_loop(loop_data, symbols, false)?;
            symbols.leave_scope();
            For(init.into(), loop_data, post)
        }
        Case(matcher, statement) => {
            Label(symbols.resolve_case(matcher)?, resolve_statement(*statement, symbols)?.into())
        }
        Default(statement) => {
            Label(symbols.resolve_default()?, resolve_statement(*statement, symbols)?.into())
        }
        Switch {
            label,
            condition,
            cases: _,
            statement,
            default: _,
        } => {
            symbols.enter_switch(label);
            let condition = resolve_expression(condition, symbols)?;
            let statement = resolve_statement(*statement, symbols)?.into();
            let switch_symbols = symbols.leave_switch();
            Switch {
                label: switch_symbols.label,
                condition,
                cases: switch_symbols.cases,
                statement,
                default: switch_symbols.default,
            }
        }
        Null => Null,
    };
    Ok(new_statement)
}

fn resolve_declaration(
    decl: Declaration,
    symbols: &mut SymbolTable,
    global: bool,
) -> Result<Declaration> {
    let decl = match decl {
        Declaration::Variable(mut var_decl) => {
            let location = var_decl.location;
            if global {
                symbols.declare_global_variable(var_decl.name.clone());
            } else {
                var_decl = symbols.declare_local_variable(var_decl).add_location(location)?;
            }
            var_decl.init =
                resolve_initializer(var_decl.init, symbols).add_location(var_decl.location)?;
            Declaration::Variable(var_decl)
        }
        Declaration::Function(func_decl) => {
            let location = func_decl.location;
            Declaration::Function(resolve_function(func_decl, symbols).add_location(location)?)
        }
    };
    Ok(decl)
}

fn resolve_initializer(
    init: Option<VariableInitializer>,
    symbols: &mut SymbolTable,
) -> Result<Option<VariableInitializer>> {
    let result = match init {
        Some(VariableInitializer::SingleElem(expr)) => {
            Some(VariableInitializer::SingleElem(resolve_expression(expr, symbols)?))
        }
        Some(VariableInitializer::CompoundInit(initializers)) => {
            let mut new_initializers = Vec::new();
            for init in initializers {
                new_initializers.push(resolve_initializer(Some(init), symbols)?.expect("Is Some"));
            }
            Some(VariableInitializer::CompoundInit(new_initializers))
        }
        None => None,
    };
    Ok(result)
}

fn resolve_expression(expr: TypedExpression, symbols: &mut SymbolTable) -> Result<TypedExpression> {
    use parse_tree::Expression::*;
    let expr = expr.expr;
    let result = match expr {
        Unary(operator, expr) => Unary(operator, resolve_expression(*expr, symbols)?.into()),
        Binary(mut binexpr) => {
            binexpr.left = resolve_expression(binexpr.left, symbols)?;
            binexpr.right = resolve_expression(binexpr.right, symbols)?;
            Binary(binexpr)
        }
        Condition(mut cond) => {
            cond.condition = resolve_expression(cond.condition, symbols)?;
            cond.if_true = resolve_expression(cond.if_true, symbols)?;
            cond.if_false = resolve_expression(cond.if_false, symbols)?;
            Condition(cond)
        }
        Assignment(mut assign) => {
            assign.left = resolve_expression(assign.left, symbols)?;
            assign.right = resolve_expression(assign.right, symbols)?;
            Assignment(assign)
        }
        Variable(name) => Variable(symbols.resolve_identifier(&name)?),
        Constant(_) => expr,
        FunctionCall(name, args) => {
            let name = symbols.resolve_identifier(&name)?;
            let args: Result<Vec<TypedExpression>> =
                args.into_iter().map(|arg| resolve_expression(arg, symbols)).collect();
            FunctionCall(name, args?)
        }
        Cast(new_type, expr) => Cast(new_type, resolve_expression(*expr, symbols)?.into()),
        Dereference(inner) => Dereference(resolve_expression(*inner, symbols)?.into()),
        AddrOf(inner) => AddrOf(resolve_expression(*inner, symbols)?.into()),
        Subscript(expr, sub_expr) => Subscript(
            resolve_expression(*expr, symbols)?.into(),
            resolve_expression(*sub_expr, symbols)?.into(),
        ),
        StringLiteral(_) => expr,
        SizeOf(_) => todo!("Size of expression semantics!"),
        SizeOfT(_) => todo!("Size of type semantics!"),
    };
    // Convert to TypedExpression
    Ok(result.into())
}

fn resolve_loop(mut loop_data: Loop, symbols: &mut SymbolTable, scoped: bool) -> Result<Loop> {
    symbols.enter_loop(loop_data.label.clone());
    if scoped {
        symbols.enter_scope();
    }
    loop_data.condition = resolve_expression(loop_data.condition, symbols)?;
    loop_data.body = resolve_statement(*loop_data.body, symbols)?.into();
    symbols.leave_loop();
    if scoped {
        symbols.leave_scope();
    }
    Ok(loop_data)
}
