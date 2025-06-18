use std::fs::File;
use std::io::Write;
use std::{fs, path::PathBuf, process::Command};

pub mod codegen;
pub mod helpers;
pub mod parse;
pub mod tac_generation;
pub mod validate;

use clap::Parser;

use crate::codegen::assembly_gen::gen_assembly_tree as gen_assembly;
use crate::codegen::assembly_rewrite::rewrite_assembly;
use crate::codegen::emission::emit_program;
use crate::helpers::error::Error;
use crate::parse::lexer::lex;
use crate::parse::parser::parse;
use crate::tac_generation::generator::gen_tac_ast as gen_tac;
use crate::validate::semantics::resolve_program as validate_program;
use crate::validate::type_check::type_check_program;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// File to compile
    filename: String,
    ///Libraries to link
    #[arg(short = 'l')]
    libraries: Vec<String>,
    #[arg(short = 'L')]
    library_paths: Vec<String>,
    #[arg(long)]
    print_commands: bool,
    /// Only run the lexer
    #[arg(short, long)]
    lex: bool,
    /// Run the parser and lexer
    #[arg(short, long)]
    parse: bool,
    #[arg(long)]
    semantics: bool,
    /// Run the parser, lexer, and type checking
    #[arg(short, long)]
    validate: bool,
    /// Run the lexer, parser, and TAC generation
    #[arg(short, long)]
    tacky: bool,
    /// Run assembly generation but without rewrites
    #[arg(long)]
    no_rewrite: bool,
    /// Run the lexer, parser, and all code gen, but don't output assembly
    #[arg(short = 'g', long)]
    codegen: bool,
    /// Output assembly, but don't run linker
    #[arg(short = 'S')]
    assembler_only: bool,
    /// Output assembly and run assembler to generate object file, but not linker,
    #[arg(short = 'c')]
    compile_only: bool,
}

pub struct CompilerError {
    pub message: String,
    pub exit_code: u8,
}
trait ConvertWithExitCode<T> {
    fn or_exit_with(self, exit_code: u8) -> T;
}
impl<T, E: ToString> ConvertWithExitCode<Result<T, CompilerError>> for Result<T, E> {
    fn or_exit_with(self, exit_code: u8) -> Result<T, CompilerError> {
        self.map_err(|err| CompilerError { message: err.to_string(), exit_code })
    }
}
trait ConvertWithSourceAndExitCode<T> {
    fn or_exit_with(self, exit_code: u8, source: &str, file: &str) -> T;
}
impl<T> ConvertWithSourceAndExitCode<Result<T, CompilerError>> for Result<T, Error> {
    fn or_exit_with(self, exit_code: u8, source: &str, file: &str) -> Result<T, CompilerError> {
        self.map_err(|err| CompilerError {
            message: err.error_message(source, file),
            exit_code,
        })
    }
}

fn path_with_extension(filename: &str, extension: &str) -> Result<String, ()> {
    let mut file = PathBuf::from(&filename);
    file.set_extension(extension);
    file.to_str().map(|path| path.to_string()).ok_or(())
}

fn run_gcc(args: &[&str], print_commands: bool) -> Result<(), String> {
    let mut gcc = Command::new("gcc");
    gcc.args(args);
    if print_commands {
        println!("Running gcc: {:?}", gcc);
    }
    let status = gcc.status().map_err(|err| err.to_string())?;
    if status.success() { Ok(()) } else { Err("GCC failed!".to_string()) }
}

pub fn run_main(args: Args) -> Result<(), CompilerError> {
    // Use GCC to preprocess
    let source_file = path_with_extension(&args.filename, "i").expect(".i extension failed!");
    run_gcc(&["-E", &args.filename, "-o", &source_file], args.print_commands).or_exit_with(1)?;
    let source = fs::read_to_string(&source_file).expect("Failed to read source code!");
    // Compiler passes
    let mut token_iter = lex(&source).or_exit_with(2, &source, &args.filename)?;
    if args.lex {
        println!("Tokens:\n\n{}", token_iter);
        return Ok(());
    }
    let parser_ast = parse(&mut token_iter).or_exit_with(3, &source, &args.filename)?;
    if args.parse {
        println!("Parser AST: {:#?}", parser_ast);
        return Ok(());
    }
    let validated_ast = validate_program(parser_ast).or_exit_with(4, &source, &args.filename)?;
    if args.semantics {
        println!("Resolved AST: {:#?}", validated_ast);
        return Ok(());
    }
    let mut typed_ast =
        type_check_program(validated_ast).or_exit_with(5, &source, &args.filename)?;
    if args.validate {
        println!("Typed AST: {:#?}", typed_ast.program);
        println!("Symbols: {:#?}", typed_ast.symbols);
        return Ok(());
    }
    let tac_ast = gen_tac(typed_ast.program, &mut typed_ast.symbols);
    if args.tacky {
        println!("Tacky AST: {:#?}", tac_ast);
        println!("Tacky symbols: {:#?}", typed_ast.symbols);
        return Ok(());
    }
    let assembly_ast = gen_assembly(tac_ast, typed_ast.symbols);
    if args.no_rewrite {
        println!("Assembly AST: {:#?}", assembly_ast);
        return Ok(());
    }
    let assembly_ast = rewrite_assembly(assembly_ast);
    if args.codegen {
        println!("Assembly AST (after rewrite): {:#?}", assembly_ast);
        return Ok(());
    }
    let mut asm_code = String::new();
    emit_program(&mut asm_code, assembly_ast).or_exit_with(6)?;
    let assembly_file = path_with_extension(&args.filename, "s").expect(".s extention failed!");
    let mut file = File::create(&assembly_file).expect("Failed to create file!");
    write!(file, "{}", asm_code).expect("Failed to write to file!");
    if args.assembler_only {
        print!("{}", asm_code);
        return Ok(());
    }
    // Use GCC to assemble and/or link
    if args.compile_only {
        let object_file = path_with_extension(&args.filename, "o").expect(".o extension failed!");
        run_gcc(&[&assembly_file, "-c", "-o", &object_file], args.print_commands).or_exit_with(7)
    } else {
        let output_file =
            path_with_extension(&args.filename, "").expect("output extension failed!");
        let linker_args = [
            vec![&assembly_file, "-o", &output_file],
            args.libraries.iter().flat_map(|lib| ["-l", lib]).collect(),
            args.library_paths.iter().flat_map(|path| ["-L", path]).collect(),
        ]
        .concat();
        run_gcc(&linker_args, args.print_commands).or_exit_with(8)
    }
}
