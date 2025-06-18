use std::process::ExitCode;

use clap::Parser;
use compiler::{Args, run_main};

fn main() -> ExitCode {
    let args = Args::parse();
    if let Err(error) = run_main(args) {
        match error.exit_code {
            1 => eprintln!("GCC Preprocessing Failed: "),
            2 => eprintln!("Lexing Failed: "),
            3 => eprintln!("Parsing Failed: "),
            4 => eprintln!("Semantic Check Failed: "),
            5 => eprintln!("Type Checking Failed: "),
            6 => eprintln!("Code Emission Failed:"),
            7 => eprintln!("GCC Assembler Failed: "),
            8 => eprintln!("GCC Assemble and Link Failed!"),
            _ => eprintln!("Unknown error!"),
        }
        eprintln!("\t{}", error.message);
        ExitCode::from(error.exit_code)
    } else {
        ExitCode::SUCCESS
    }
}
