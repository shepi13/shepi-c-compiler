use std::process::exit;

use clap::Parser;
use compiler::{Args, run_main};

fn main() {
    let args = Args::parse();
    if let Err(error) = run_main(args) {
        match error.exit_code {
            1 => eprintln!("GCC Preprocessing Failed!\n"),
            2 => eprintln!("Lex Error: "),
            3 => eprintln!("Parse Error: "),
            4 => eprintln!("Semantic Error: "),
            5 => eprintln!("Type Error: "),
            6 => eprintln!("Code Emission Failed!\n"),
            7 => eprintln!("GCC Assembler Failed!\n"),
            8 => eprintln!("GCC Assemble and Link Failed!\n"),
            _ => eprintln!("Unknown error!"),
        }
        eprintln!("\t{}", error.message);
        exit(error.exit_code as i32);
    }
}
