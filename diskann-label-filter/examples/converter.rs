/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::{File, OpenOptions},
    io::{BufRead, BufReader, Write},
};

/// Converts a base file from old comma-separated format to new JSON format
///
/// # Arguments
/// * `input_path` - Path to the input file with comma-separated key=value pairs
/// * `output_path` - Path to the output JSON file
///
/// # Example
/// Input: "year=2007,month=August,camera=OLYMPUS"
/// Output: {"id": 0, "year": "2007", "month": "August", "camera": "OLYMPUS"}
pub fn convert_base_file(
    input_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_file = File::open(input_path)?;
    let reader = BufReader::new(input_file);

    let mut output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        let json_entry = parse_base_line(line, line_num)?;
        writeln!(output_file, "{}", json_entry)?;
    }

    println!("Successfully converted {} to {}", input_path, output_path);
    Ok(())
}

/// Converts a query file from old ampersand-separated format to new JSON format
///
/// # Arguments
/// * `input_path` - Path to the input file with ampersand-separated key=value pairs
/// * `output_path` - Path to the output JSON file
///
/// # Example
/// Input: "camera=SONY&year=2007"
/// Output: {"query_id": 0, "filter": {"$and": [{"camera": {"$eq": "SONY"}}, {"year": {"$eq": 2007}}]}}
pub fn convert_query_file(
    input_path: &str,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let input_file = File::open(input_path)?;
    let reader = BufReader::new(input_file);

    let mut output_file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(output_path)?;

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        let json_entry = parse_query_line(line, line_num)?;
        writeln!(output_file, "{}", json_entry)?;
    }

    println!("Successfully converted {} to {}", input_path, output_path);
    Ok(())
}

/// Parses a single line with comma-separated key=value pairs into JSON format
fn parse_base_line(line: &str, line_num: usize) -> Result<String, Box<dyn std::error::Error>> {
    let mut json_fields = Vec::new();

    // Add the id field first
    json_fields.push(format!("\"id\": {}", line_num));

    // Parse key=value pairs separated by commas
    for pair in line.split(',') {
        let parts: Vec<&str> = pair.split('=').collect();
        if parts.len() != 2 {
            continue; // Skip malformed pairs
        }

        let key = parts[0].trim();
        let value = parts[1].trim();

        // Escape quotes in the value and format as JSON
        let escaped_value = escape_json_string(value);
        json_fields.push(format!("\"{}\": \"{}\"", key, escaped_value));
    }

    Ok(format!("{{{}}}", json_fields.join(", ")))
}

/// Parses a single line with ampersand-separated key=value pairs into JSON query format
fn parse_query_line(line: &str, line_num: usize) -> Result<String, Box<dyn std::error::Error>> {
    let mut filters = Vec::new();

    // Parse key=value pairs separated by ampersands
    for pair in line.split('&') {
        let parts: Vec<&str> = pair.split('=').collect();
        if parts.len() != 2 {
            continue; // Skip malformed pairs
        }

        let key = parts[0].trim();
        let value = parts[1].trim();

        // Try to parse value as number, otherwise treat as string
        let filter_value = if let Ok(num_val) = value.parse::<i64>() {
            format!("\"{}\"", num_val)
        } else {
            format!("\"{}\"", escape_json_string(value))
        };

        // Create filter condition
        let filter_condition = format!("{{\"{}\": {{\"$eq\": {}}}}}", key, filter_value);
        filters.push(filter_condition);
    }

    // Build the complete query JSON
    let query_json = match filters.len() {
        1 => format!("{{\"query_id\": {}, \"filter\": {}}}", line_num, filters[0]),
        _ => {
            let filter_array = format!("[{}]", filters.join(", "));
            format!(
                "{{\"query_id\": {}, \"filter\": {{\"$and\": {}}}}}",
                line_num, filter_array
            )
        }
    };

    Ok(query_json)
}

/// Escapes special characters in JSON strings
fn escape_json_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    fn get_temp_dir() -> PathBuf {
        PathBuf::from("test_temp")
    }

    #[test]
    fn test_parse_base_line() {
        let line = "year=2007,month=August,camera=OLYMPUS";
        let result = parse_base_line(line, 0).unwrap();

        assert!(result.contains("\"id\": 0"));
        assert!(result.contains("\"year\": \"2007\""));
        assert!(result.contains("\"month\": \"August\""));
        assert!(result.contains("\"camera\": \"OLYMPUS\""));
    }

    #[test]
    fn test_parse_query_line() {
        let line = "camera=SONY&year=2007";
        let result = parse_query_line(line, 0).unwrap();

        assert!(result.contains("\"query_id\": 0"));
        assert!(result.contains("\"$and\""));
        assert!(result.contains("\"camera\": {\"$eq\": \"SONY\"}"));
        assert!(result.contains("\"year\": {\"$eq\": \"2007\"}"));
    }

    #[test]
    fn test_convert_base_file() {
        let input_path = "test_base_input.txt";
        let output_path = "test_base_output.json";

        // Create test input file
        let input_content =
            "year=2007,month=August,camera=OLYMPUS\nyear=2008,month=July,camera=CANON";
        fs::write(input_path, input_content).unwrap();

        // Convert the file
        convert_base_file(input_path, output_path).unwrap();

        // Verify output file was created and has correct content
        assert!(std::path::Path::new(output_path).exists());
        let output_content = fs::read_to_string(output_path).unwrap();

        assert!(output_content.contains("\"id\": 0"));
        assert!(output_content.contains("\"year\": \"2007\""));
        assert!(output_content.contains("\"id\": 1"));
        assert!(output_content.contains("\"year\": \"2008\""));

        // Clean up
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_convert_query_file() {
        let input_path = "test_query_input.txt";
        let output_path = "test_query_output.json";

        // Create test input file
        let input_content = "camera=SONY&year=2007\ncategory=laptop&price=1500";
        fs::write(input_path, input_content).unwrap();

        // Convert the file
        convert_query_file(input_path, output_path).unwrap();

        // Verify output file was created and has correct content
        assert!(std::path::Path::new(output_path).exists());
        let output_content = fs::read_to_string(output_path).unwrap();

        assert!(output_content.contains("\"query_id\": 0"));
        assert!(output_content.contains("\"$and\""));
        assert!(output_content.contains("\"camera\": {\"$eq\": \"SONY\"}"));
        assert!(output_content.contains("\"query_id\": 1"));
        assert!(output_content.contains("\"category\": {\"$eq\": \"laptop\"}"));

        // Clean up
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn test_escape_json_string() {
        assert_eq!(escape_json_string("normal"), "normal");
        assert_eq!(escape_json_string("with\"quote"), "with\\\"quote");
        assert_eq!(escape_json_string("with\\slash"), "with\\\\slash");
    }
}

/// Main function for command line usage
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() != 5 {
        eprintln!(
            "Usage: {} <base_input_file> <query_input_file> <base_output_file> <query_output_file>",
            args[0]
        );
        eprintln!(
            "Example: {} old_base.txt old_query.txt new_base.json new_query.json",
            args[0]
        );
        std::process::exit(1);
    }

    let base_input_path = &args[1];
    let query_input_path = &args[2];
    let base_output_path = &args[3];
    let query_output_path = &args[4];

    // Convert both files
    convert_base_file(base_input_path, base_output_path)?;
    convert_query_file(query_input_path, query_output_path)?;

    println!("Conversion completed successfully!");

    Ok(())
}
