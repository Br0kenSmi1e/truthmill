use truthmill::{Problem, log_flops, symbolic_flops, validate, verify};

const PROBLEMS: &[(&str, &str)] = &[
    ("MP3", include_str!("../examples/chemistry/mp3.json")),
    ("CCD", include_str!("../examples/chemistry/ccd.json")),
    ("CCSD", include_str!("../examples/chemistry/ccsd.json")),
    ("CCSD(T)", include_str!("../examples/chemistry/ccsd_t.json")),
    (
        "EOM-CCSD",
        include_str!("../examples/chemistry/eom_ccsd.json"),
    ),
    ("ADC(2)", include_str!("../examples/chemistry/adc2.json")),
    (
        "CCSD Lambda",
        include_str!("../examples/chemistry/ccsd_lambda.json"),
    ),
];

#[test]
fn chemistry_problems_are_valid_and_scorable() {
    for &(name, json) in PROBLEMS {
        let problem = Problem::from_json(json)
            .unwrap_or_else(|error| panic!("failed to deserialize {name}: {error}"));
        validate(&problem).unwrap_or_else(|error| panic!("invalid {name}: {error}"));
        log_flops(&problem, &problem.reference)
            .unwrap_or_else(|error| panic!("failed to score {name}: {error}"));
        let symbolic = symbolic_flops(&problem, &problem.reference)
            .unwrap_or_else(|error| panic!("failed to symbolically score {name}: {error}"));
        assert!(!symbolic.is_zero(), "{name} has zero symbolic FLOPs");
    }
}

#[test]
fn chemistry_references_verify_against_themselves() {
    for &(name, json) in PROBLEMS {
        let problem = Problem::from_json(json).unwrap();
        assert_eq!(
            verify(&problem, &problem.reference),
            Ok(true),
            "{name} reference did not verify"
        );
    }
}
