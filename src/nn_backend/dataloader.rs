use std::{fs, path::Path};

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, InMemDataset},
    },
    serde,
    tensor::{backend::Backend, ElementConversion},
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
    pub overwrite_buf: Option<Vec<u8>>,
    pub buf_len: Option<usize>,
    pub overwrite_offset: Option<usize>,
    pub dataset: InMemDataset<DataItem>,
    pub max_size: Option<usize>,
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
    pub fn retrain<B: Backend>(
        c_dataset: CustomDataset,
        device: &B::Device,
        model: crate::model::Model<B>,
    ) -> Self {
        let mut dataset: Vec<u8> = Vec::with_capacity(c_dataset.max_size.unwrap_or(0));
        let batcher = super::batcher::InternalBatcher::<B>::new(device.clone());
        let batch = batcher.batch(
            (0..c_dataset.max_size.unwrap_or(0))
                .collect::<std::vec::Vec<usize>>()
                .iter()
                .enumerate()
                .map(|(i, _v)| DataItem {
                    address: i as u64,
                    value: 0u8,
                })
                .collect(),
        );
        let out: Vec<_> = model.forward(batch.addresses).into_data().value;
        out.iter()
            .enumerate()
            .for_each(|(i, v)| dataset[i] = v.elem::<f32>() as u8);
        let dataset = InMemDataset::new(
            dataset
                .iter()
                .enumerate()
                .map(|(i, v)| DataItem {
                    address: i as u64,
                    value: *v,
                })
                .collect(),
        );
        CustomDataset {
            overwrite_buf: c_dataset.overwrite_buf,
            buf_len: c_dataset.buf_len,
            overwrite_offset: c_dataset.overwrite_offset,
            dataset,
            max_size: c_dataset.max_size,
        }
    }
    pub fn new(max_size: usize) -> Self {
        let mut dataset: Vec<u8> = Vec::with_capacity(max_size);
        dataset = dataset.iter().map(|_| 0).collect();
        Self {
            overwrite_buf: None,
            buf_len: None,
            overwrite_offset: None,
            max_size: None,
            dataset: InMemDataset::new(
                dataset
                    .iter()
                    .enumerate()
                    .map(|(i, v)| DataItem {
                        address: i as u64,
                        value: *v,
                    })
                    .collect(),
            ),
        }
    }
    pub fn old_new() -> Self {
        let mut dataset: Vec<u8> = Vec::new();
        let contents = fs::read_to_string(Path::new("./tmp/copypasta.csv")).unwrap();
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
            max_size: None,
        }
    }
}

impl Default for CustomDataset {
    fn default() -> Self {
        Self::old_new()
    }
}
