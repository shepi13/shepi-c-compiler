pub fn unescape_char(value: &str) -> Result<i32, String> {
    let mut char_iter = value.chars();
    match char_iter.next() {
        Some('\\') => {
            let next_char = char_iter.next().ok_or("Missing escape sequence!")?;
            Ok(_unescape_char_code(next_char)?)
        }
        Some(val) => Ok(val as i32),
        None => Err("Empty character literal!".into()),
    }
}

pub fn unescape_string(value: &str) -> Result<String, String> {
    let mut chars = value.chars();
    let mut result = String::new();
    while let Some(current_char) = chars.next() {
        if current_char == '\\' {
            let escape_seq = chars.next().ok_or("Missing escape sequence!")?;
            let char_code = _unescape_char_code(escape_seq)?;
            result.push(char::from_u32(char_code as u32).expect("Is single byte!"));
        } else {
            result.push(current_char);
        }
    }
    Ok(result)
}

fn _unescape_char_code(val: char) -> Result<i32, String> {
    let result = match val {
        '\'' => 39,
        '\"' => 34,
        '?' => 63,
        '\\' => 92,
        'a' => 7,
        'b' => 8,
        't' => 9,
        'n' => 10,
        'v' => 11,
        'f' => 12,
        'r' => 13,
        _ => return Err("Unrecognized escape sequence!".into()),
    };
    Ok(result)
}

pub fn escape_asm_string(data: String) -> String {
    let new_string = data.replace("\\", "\\134");
    let new_string = new_string.replace("\n", "\\12");
    new_string.replace("\"", "\\42")
}
