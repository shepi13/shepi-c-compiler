use std::collections::HashMap;

use crate::{
    parser::{
        Block, BlockItem, CType, Declaration, Expression, ForInit, FunctionDeclaration, Program,
        Statement, StorageClass, VariableDeclaration,
    },
    semantics::eval_constant_expr,
};

pub type Symbols = HashMap<String, Symbol>;
#[derive(Debug)]
pub struct Symbol {
    pub ctype: CType,
    pub attrs: SymbolAttr,
}

impl Symbol {
    pub fn get_function_attrs(&self) -> &FunctionAttributes {
        match &self.attrs {
            SymbolAttr::FUNCTION(attrs) => attrs,
            _ => panic!("Not a function!"),
        }
    }
    pub fn get_static_attrs(&self) -> &StaticAttributes {
        match &self.attrs {
            SymbolAttr::STATIC(attrs) => attrs,
            _ => panic!("Not a static variable: {:#?}", self),
        }
    }
}

#[derive(Debug)]
pub enum SymbolAttr {
    FUNCTION(FunctionAttributes),
    STATIC(StaticAttributes),
    LOCAL,
}

#[derive(Debug)]
pub struct FunctionAttributes {
    pub defined: bool,
    pub global: bool,
}
#[derive(Debug)]
pub struct StaticAttributes {
    pub init: StaticInitializer,
    pub global: bool,
}
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum StaticInitializer {
    TENTATIVE,
    INITIALIZER(i32),
    NONE,
}

// Tree traversal functions
pub fn type_check_program(program: &Program) -> Symbols {
    let mut symbols = HashMap::new();
    for decl in program {
        type_check_declaration(decl, &mut symbols, true);
    }
    symbols
}

fn type_check_declaration(decl: &Declaration, symbols: &mut Symbols, global: bool) {
    match decl {
        Declaration::FUNCTION(func) => type_check_function(func, symbols),
        Declaration::VARIABLE(var) => {
            if global {
                type_check_var_filescope(var, symbols);
            } else {
                type_check_var_declaration(var, symbols);
            }
        }
    }
}

fn type_check_block(block: &Block, symbols: &mut Symbols) {
    for block_item in block {
        type_check_block_item(block_item, symbols);
    }
}
fn type_check_block_item(block_item: &BlockItem, symbols: &mut Symbols) {
    match block_item {
        BlockItem::STATEMENT(statement) => type_check_statement(statement, symbols),
        BlockItem::DECLARATION(decl) => type_check_declaration(decl, symbols, false),
    }
}
fn type_check_statement(statement: &Statement, symbols: &mut Symbols) {
    match statement {
        Statement::RETURN(expr) | Statement::EXPRESSION(expr) => {
            type_check_expression(expr, symbols);
        }
        Statement::LABEL(_, statement)
        | Statement::DEFAULT(statement)
        | Statement::CASE(_, statement) => {
            type_check_statement(statement, symbols);
        }
        Statement::IF(expr, if_true, if_false) => {
            type_check_expression(expr, symbols);
            type_check_statement(if_true, symbols);
            if let Some(false_statement) = &**if_false {
                type_check_statement(false_statement, symbols);
            }
        }
        Statement::WHILE(loop_data) | Statement::DOWHILE(loop_data) => {
            type_check_expression(&loop_data.condition, symbols);
            type_check_statement(&loop_data.body, symbols);
        }
        Statement::COMPOUND(block) => {
            type_check_block(block, symbols);
        }
        Statement::SWITCH(switch) => {
            type_check_expression(&switch.condition, symbols);
            type_check_statement(&switch.statement, symbols)
        }
        Statement::FOR(init, loop_data, post) => {
            match init {
                ForInit::INITDECL(decl) => type_check_var_declaration(decl, symbols),
                ForInit::INITEXP(Some(expr)) => type_check_expression(expr, symbols),
                _ => (),
            }
            type_check_expression(&loop_data.condition, symbols);
            if let Some(post_expr) = post {
                type_check_expression(post_expr, symbols);
            }
            type_check_statement(&loop_data.body, symbols);
        }
        Statement::BREAK(_) | Statement::CONTINUE(_) | Statement::GOTO(_) | Statement::NULL => (),
    }
}

fn type_check_expression(expr: &Expression, symbols: &mut Symbols) {
    match expr {
        Expression::UNARY(_, expr) => type_check_expression(expr, symbols),
        Expression::VAR(name) => type_check_variable(&name, symbols),
        Expression::FUNCTION(name, args) => type_check_call(&name, args, symbols),
        Expression::BINARY(exprs) => {
            type_check_expression(&exprs.left, symbols);
            type_check_expression(&exprs.right, symbols);
        }
        Expression::ASSIGNMENT(exprs) => {
            type_check_expression(&exprs.left, symbols);
            type_check_expression(&exprs.right, symbols);
        }
        Expression::CONDITION(cond) => {
            type_check_expression(&cond.condition, symbols);
            type_check_expression(&cond.if_true, symbols);
            type_check_expression(&cond.if_false, symbols);
        }
        Expression::LITEXP(_) => (),
    }
}

// Type Checking

fn type_check_function(function: &FunctionDeclaration, symbols: &mut Symbols) {
    let ctype = CType::FUNCTION(function.params.len());
    let mut defined = function.body.is_some();
    let mut global = function.storage != Some(StorageClass::STATIC);
    if let Some(symbol) = symbols.get(&function.name) {
        let attrs = symbol.get_function_attrs();
        assert!(symbol.ctype == ctype, "Incompatible function declarations!");
        assert!(!attrs.defined || !defined, "Function redefinition!");
        assert!(
            !attrs.global || global,
            "static definition cannot follow non-static"
        );
        global = attrs.global;
        defined = attrs.defined;
    }
    symbols.insert(
        function.name.clone(),
        Symbol {
            ctype,
            attrs: SymbolAttr::FUNCTION(FunctionAttributes { defined, global }),
        },
    );
    if let Some(body) = &function.body {
        for param in &function.params {
            symbols.insert(
                param.clone(),
                Symbol {
                    ctype: CType::INT,
                    attrs: SymbolAttr::LOCAL,
                },
            );
        }
        type_check_block(body, symbols);
    }
}

fn type_check_call(name: &str, args: &Vec<Expression>, symbols: &mut Symbols) {
    for arg in args {
        type_check_expression(arg, symbols);
    }
    assert!(
        symbols[name].ctype == CType::FUNCTION(args.len()),
        "Function call has invalid type"
    );
}

fn type_check_var_declaration(var: &VariableDeclaration, symbols: &mut Symbols) {
    match var.storage {
        Some(StorageClass::EXTERN) => {
            assert!(
                var.value.is_none(),
                "Local extern variable cannot have initializer!"
            );
            if let Some(symbol) = symbols.get(&var.name) {
                assert!(
                    symbol.ctype == CType::INT,
                    "Function redeclared as variable!"
                );
            } else {
                symbols.insert(
                    var.name.clone(),
                    Symbol {
                        ctype: var.ctype.clone(),
                        attrs: SymbolAttr::STATIC(StaticAttributes {
                            init: StaticInitializer::NONE,
                            global: true,
                        }),
                    },
                );
            }
        }
        Some(StorageClass::STATIC) => {
            let init_value = if var.value.is_none() {
                StaticInitializer::INITIALIZER(0)
            } else {
                StaticInitializer::INITIALIZER(eval_constant_expr(var.value.as_ref().unwrap()))
            };
            symbols.insert(
                var.name.clone(),
                Symbol {
                    ctype: var.ctype.clone(),
                    attrs: SymbolAttr::STATIC(StaticAttributes {
                        init: init_value,
                        global: false,
                    }),
                },
            );
        }
        _ => {
            symbols.insert(
                var.name.clone(),
                Symbol {
                    ctype: var.ctype.clone(),
                    attrs: SymbolAttr::LOCAL,
                },
            );
            if let Some(init) = &var.value {
                type_check_expression(init, symbols);
            }
        }
    }
}

fn type_check_var_filescope(var: &VariableDeclaration, symbols: &mut Symbols) {
    let mut init_value = if var.value.is_none() {
        if var.storage == Some(StorageClass::EXTERN) {
            StaticInitializer::NONE
        } else {
            StaticInitializer::TENTATIVE
        }
    } else {
        StaticInitializer::INITIALIZER(eval_constant_expr(var.value.as_ref().unwrap()))
    };

    let mut global = var.storage != Some(StorageClass::STATIC);
    if let Some(symbol) = symbols.get(&var.name) {
        let attrs = symbol.get_static_attrs();
        assert!(
            symbol.ctype == CType::INT,
            "Function redeclared as variable!"
        );
        if var.storage == Some(StorageClass::EXTERN) {
            global = attrs.global;
        } else if attrs.global != global {
            panic!("Conflicting linkage!");
        }

        if matches!(attrs.init, StaticInitializer::INITIALIZER(_)) {
            assert!(
                !matches!(init_value, StaticInitializer::INITIALIZER(_)),
                "Conflicting static initializers!"
            );
            init_value = attrs.init.clone();
        } else if attrs.init == StaticInitializer::TENTATIVE
            && !matches!(init_value, StaticInitializer::INITIALIZER(_))
        {
            init_value = StaticInitializer::TENTATIVE
        }
    }
    symbols.insert(
        var.name.clone(),
        Symbol {
            ctype: var.ctype.clone(),
            attrs: SymbolAttr::STATIC(StaticAttributes {
                init: init_value,
                global,
            }),
        },
    );
}

fn type_check_variable(name: &str, symbols: &mut Symbols) {
    assert!(
        symbols[name].ctype == CType::INT,
        "Found function, not variable!"
    );
}
