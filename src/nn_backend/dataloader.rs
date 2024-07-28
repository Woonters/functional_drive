use std::{fs, path::Path};

use burn::{
    data::dataset::{Dataset, InMemDataset},
    serde,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DataItem {
    // Memory Location
    #[serde(rename = "Address")]
    pub address: u64,
    // Value
    #[serde(rename = "Value")]
    pub value: u8,
}

pub struct CustomDataset {
    overwrite_buf: Option<Vec<u8>>,
    buf_len: Option<usize>,
    overwrite_offset: Option<usize>,
    dataset: InMemDataset<DataItem>,
}

impl Dataset<DataItem> for CustomDataset {
    fn get(&self, index: usize) -> Option<DataItem> {
        if self.overwrite_buf.is_some() {
            if index < self.overwrite_offset.unwrap()
                || index >= self.overwrite_offset.unwrap() + self.buf_len.unwrap()
            {
                self.dataset.get(index)
            } else {
                return Some(DataItem {
                    address: index as u64,
                    value: self.overwrite_buf.as_ref().unwrap()
                        [index - self.overwrite_offset.unwrap()],
                });
            }
        } else {
            self.dataset.get(index)
        }
    }
    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl CustomDataset {
    pub fn new() -> Self {
        let mut dataset: Vec<u8> = Vec::new();
        let contents = fs::read_to_string(Path::new("./copypasta.csv")).unwrap();
        contents
            .split("\n")
            .for_each(|v| dataset.push(v.parse::<u8>().unwrap_or_default()));
        let dataset: InMemDataset<DataItem> = InMemDataset::new(
            dataset
                .iter()
                .enumerate()
                .map(|(i, v)| DataItem {
                    address: i as u64,
                    value: *v,
                })
                .collect(),
        );
        Self {
            overwrite_buf: None,
            buf_len: None,
            overwrite_offset: None,
            dataset,
        }
    }
}

impl Default for CustomDataset {
    fn default() -> Self {
        Self::new()
    }
}
