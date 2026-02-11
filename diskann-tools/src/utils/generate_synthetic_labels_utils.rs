/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{collections::HashMap, fs::File, io::Write};

use diskann::ANNResult;
use rand::{
    distr::{Bernoulli, Distribution},
    Rng,
};
use tracing::{error, info};

struct ZipfDistribution {
    num_labels: u32,
    num_points: usize,
    distribution_factor: f64,
}

impl ZipfDistribution {
    fn new(num_points: usize, num_labels: u32) -> Self {
        ZipfDistribution {
            num_labels,
            num_points,
            distribution_factor: 0.7,
        }
    }

    fn create_distribution_map(&self) -> HashMap<u32, u32> {
        let mut map = HashMap::new();
        let primary_label_freq = (self.num_points as f64 * self.distribution_factor) as u32;

        for i in 1..=self.num_labels {
            map.insert(i, primary_label_freq / i);
        }

        map
    }

    fn write_distribution<W: Write>(&mut self, mut file: W, rng: &mut impl Rng) -> ANNResult<()> {
        let mut distribution_map = self.create_distribution_map();

        for i in 0..self.num_points {
            let mut label_written = false;

            for (label, freq) in distribution_map.iter_mut() {
                let label_selection_probability =
                    Bernoulli::new(self.distribution_factor / (*label as f64));

                match label_selection_probability {
                    Ok(bernoulli) => {
                        if bernoulli.sample(rng) && *freq > 0 {
                            if label_written {
                                write!(file, ",")?;
                            }
                            write!(file, "{}", label)?;
                            label_written = true;
                            // Remove label from map if we have used all labels
                            *freq -= 1;
                        }
                    }
                    Err(err) => {
                        error!("Error creating Bernoulli distribution: {:?}", err);
                    }
                }
            }

            if !label_written {
                write!(file, "0")?;
            }

            if i < self.num_points - 1 {
                writeln!(file)?;
            }
        }

        Ok(())
    }
}

pub fn generate_labels(
    output_file: &str,
    distribution_type: &str,
    num_points: usize,
    num_labels: u32,
) -> ANNResult<()> {
    let mut file = File::create(output_file)?;

    let rng = &mut diskann_providers::utils::create_rnd_from_seed(42);
    if distribution_type == "zipf" {
        let mut zipf = ZipfDistribution::new(num_points, num_labels);
        zipf.write_distribution(file, rng)?;
    } else if distribution_type == "random" {
        for i in 0..num_points {
            let mut label_written = false;
            for j in 1..=num_labels {
                // 50% chance to assign each label
                if rng.random::<bool>() {
                    if label_written {
                        write!(file, ",")?;
                    }
                    write!(file, "{}", j)?;
                    label_written = true;
                }
            }
            if !label_written {
                write!(file, "0")?;
            }
            if i < num_points - 1 {
                writeln!(file)?;
            }
        }
    } else if distribution_type == "one_per_point" {
        for i in 0..num_points {
            let lbl = rng.random_range(0..num_labels);
            write!(file, "{}", lbl)?;
            if i != num_points - 1 {
                writeln!(file)?;
            }
        }
    }

    info!("Labels written to {}.", output_file);

    Ok(())
}

#[cfg(test)]
mod test {
    use std::fs;
    use std::io::BufRead;

    use super::generate_labels;

    #[test]
    fn generate_label_test() {
        let label_file1: &str = "rand_labels_50_10K_zipf.txt";
        let _ = generate_labels(label_file1, "zipf", 10000, 50);

        assert!(
            fs::metadata(label_file1).is_ok(),
            "zipf file not found: {}",
            label_file1
        );

        let label_file2: &str = "rand_labels_50_10K_random.txt";
        let _ = generate_labels(label_file2, "random", 10000, 50);

        assert!(
            fs::metadata(label_file2).is_ok(),
            "random file not found: {}",
            label_file2
        );

        let label_file3: &str = "rand_labels_50_10K_one_per_point.txt";
        let _ = generate_labels(label_file3, "one_per_point", 10000, 50);

        assert!(
            fs::metadata(label_file3).is_ok(),
            "one_per_point file not found: {}",
            label_file3
        );

        fs::remove_file(label_file1).expect("Failed to delete file");
        fs::remove_file(label_file2).expect("Failed to delete file");
        fs::remove_file(label_file3).expect("Failed to delete file");
    }

    #[test]
    fn test_generate_labels_small_dataset() {
        let label_file = "/tmp/test_labels_small.txt";
        let result = generate_labels(label_file, "zipf", 10, 5);
        
        assert!(result.is_ok());
        assert!(fs::metadata(label_file).is_ok());
        
        // Verify we have 10 lines
        let file = fs::File::open(label_file).unwrap();
        let reader = std::io::BufReader::new(file);
        let lines: Vec<_> = reader.lines().collect();
        assert_eq!(lines.len(), 10);
        
        fs::remove_file(label_file).ok();
    }

    #[test]
    fn test_generate_labels_random_distribution() {
        let label_file = "/tmp/test_labels_random.txt";
        let result = generate_labels(label_file, "random", 100, 10);
        
        assert!(result.is_ok());
        assert!(fs::metadata(label_file).is_ok());
        
        fs::remove_file(label_file).ok();
    }

    #[test]
    fn test_generate_labels_one_per_point() {
        let label_file = "/tmp/test_labels_one_per_point.txt";
        let result = generate_labels(label_file, "one_per_point", 50, 20);
        
        assert!(result.is_ok());
        assert!(fs::metadata(label_file).is_ok());
        
        // Verify we have 50 lines
        let file = fs::File::open(label_file).unwrap();
        let reader = std::io::BufReader::new(file);
        let lines: Vec<_> = reader.lines().collect();
        assert_eq!(lines.len(), 50);
        
        fs::remove_file(label_file).ok();
    }

    #[test]
    fn test_generate_labels_single_point() {
        let label_file = "/tmp/test_labels_single.txt";
        let result = generate_labels(label_file, "zipf", 1, 5);
        
        assert!(result.is_ok());
        assert!(fs::metadata(label_file).is_ok());
        
        fs::remove_file(label_file).ok();
    }
}
