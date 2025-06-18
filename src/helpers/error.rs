use owo_colors::OwoColorize;

use crate::parse::parse_tree::Location;
use std::fmt::Write;

#[derive(Debug)]
pub struct Error {
    location: Option<Location>,
    message: String,
    error_type: String,
}
type Result<T> = std::result::Result<T, Error>;
pub fn assert_or_err(assertion: bool, error_type: &str, message: &str) -> Result<()> {
    if assertion { Ok(()) } else { Err(Error::new(error_type, message)) }
}
impl Error {
    pub fn new(error_type: &str, msg: &str) -> Self {
        Self {
            location: None,
            error_type: error_type.to_string(),
            message: msg.to_string(),
        }
    }
    pub fn error_message(&self, source: &str, file: &str) -> String {
        let mut result = String::new();
        writeln!(result, "{}: {}", self.error_type.bright_red(), self.message).unwrap();
        let lines: Vec<&str> = source.lines().filter(|line| !line.starts_with("#")).collect();
        if let Some(loc) = self.location {
            writeln!(result, "At {}:{}:{}", file, loc.start_loc.0 + 1, loc.start_loc.1 + 1)
                .unwrap();
            writeln!(result, "{}", &lines[loc.start_loc.0]).unwrap();
            for _ in 0..loc.start_loc.1 {
                write!(result, " ").unwrap();
            }
            writeln!(result, "{}", "^".green()).unwrap();
            for line in &lines[loc.start_loc.0 + 1..=loc.end_loc.0] {
                writeln!(result, "{}", line).unwrap();
            }
        }
        result.trim().to_string()
    }
    pub fn add_location(&mut self, location: Location) {
        self.location = Some(self.location.unwrap_or(location));
    }
}
pub trait AddLocation {
    fn add_location(self, location: Location) -> Self;
}
impl<T> AddLocation for Result<T> {
    fn add_location(self, location: Location) -> Self {
        match self {
            Ok(val) => Ok(val),
            Err(mut error) => {
                error.add_location(location);
                Err(error)
            }
        }
    }
}
