/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::path::Path;

use super::*;
use crate::test::TestConfig;

#[test]
fn overwrite_accepts_only_the_documented_value() {
    assert!(!parse_overwrite(Err(std::env::VarError::NotPresent)).unwrap());
    assert!(parse_overwrite(Ok("overwrite".to_owned())).unwrap());
    assert!(parse_overwrite(Ok("other".to_owned())).is_err());
}

#[test]
fn resolve_expands_every_fixture_token() {
    let input = Path::new("input.json");
    let dir = Path::new("fixture");
    let temp = Path::new("temp");
    let cases = [
        ("$INPUT", input.to_path_buf()),
        ("$OUTPUT", temp.join(OUTPUT_FILE)),
        ("$SETUP_OUTPUT", temp.join("setup-output.json")),
        ("$TOLERANCES", dir.join(TOLERANCES_FILE)),
        ("$REGRESSION_INPUT", dir.join(REGRESSION_INPUT_FILE)),
        ("$CHECK_OUTPUT", temp.join(CHECK_OUTPUT_FILE)),
        ("literal", PathBuf::from("literal")),
    ];
    for (value, expected) in cases {
        assert_eq!(resolve(value, input, dir, temp), expected);
    }
}

#[test]
fn materialize_input_rewrites_each_supported_placeholder() {
    let fixture = tempfile::tempdir().unwrap();
    let output = tempfile::tempdir().unwrap();
    let input = fixture.path().join(INPUT_FILE);

    assert_eq!(
        materialize_input(fixture.path(), output.path()).unwrap(),
        input
    );

    std::fs::write(&input, r#"{"path":"literal"}"#).unwrap();
    assert_eq!(
        materialize_input(fixture.path(), output.path()).unwrap(),
        input
    );

    for placeholder in [WORKSPACE, "$TEMPDIR"] {
        std::fs::write(&input, format!(r#"{{"path":"{placeholder}"}}"#)).unwrap();
        let materialized = materialize_input(fixture.path(), output.path()).unwrap();
        assert_eq!(materialized, output.path().join(INPUT_FILE));
        assert!(!std::fs::read_to_string(materialized)
            .unwrap()
            .contains(placeholder));
    }
}

#[test]
fn features_reads_optional_nonempty_lines() {
    let fixture = tempfile::tempdir().unwrap();
    assert!(features(fixture.path()).unwrap().is_empty());
    std::fs::write(fixture.path().join(FEATURES_FILE), "one\n\n two \n").unwrap();
    assert_eq!(features(fixture.path()).unwrap(), ["one", "two"]);

    std::fs::remove_file(fixture.path().join(FEATURES_FILE)).unwrap();
    std::fs::create_dir(fixture.path().join(FEATURES_FILE)).unwrap();
    assert!(features(fixture.path()).is_err());
}

#[test]
fn run_reports_output_mismatch() {
    let fixture = tempfile::tempdir().unwrap();
    std::fs::write(fixture.path().join(STDIN), "inputs test-input-dim\n").unwrap();
    std::fs::write(fixture.path().join(STDOUT), "incorrect output\n").unwrap();
    let mut registry = Registry::new();
    crate::test::register_benchmarks(&mut registry, &TestConfig::new()).unwrap();

    assert!(run(fixture.path(), &registry).is_err());
}

#[test]
fn generated_output_contract_handles_every_presence_state() {
    let fixture = tempfile::tempdir().unwrap();
    let generated = tempfile::tempdir().unwrap();
    let filename = "result.json";
    let expected = fixture.path().join(filename);
    let actual = generated.path().join(filename);

    std::fs::write(&actual, "new").unwrap();
    check_generated_output(fixture.path(), generated.path(), filename, true).unwrap();
    assert_eq!(std::fs::read_to_string(&expected).unwrap(), "new");

    std::fs::remove_file(&actual).unwrap();
    check_generated_output(fixture.path(), generated.path(), filename, true).unwrap();
    assert!(!expected.exists());

    std::fs::write(&actual, "unexpected").unwrap();
    assert!(check_generated_output(fixture.path(), generated.path(), filename, false).is_err());
    std::fs::remove_file(&actual).unwrap();

    std::fs::write(&expected, "expected").unwrap();
    assert!(check_generated_output(fixture.path(), generated.path(), filename, false).is_err());

    std::fs::write(&actual, "different").unwrap();
    assert!(check_generated_output(fixture.path(), generated.path(), filename, false).is_err());
    std::fs::write(&actual, "expected").unwrap();
    check_generated_output(fixture.path(), generated.path(), filename, false).unwrap();
}

#[test]
fn parse_apps_rejects_comment_only_fixture() {
    let fixture = tempfile::tempdir().unwrap();
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(fixture.path().join(STDIN), "# comment\n\n").unwrap();
    assert!(parse_apps(fixture.path(), Path::new("input"), temp.path()).is_err());
}
