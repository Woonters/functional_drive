use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Tensor},
};

use super::dataloader::DataItem;

#[derive(Clone)]
pub struct InternalBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> InternalBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct Batch<B: Backend> {
    pub addresses: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

fn u64_to_bits(input: u64) -> [f32; 64] {
    let out: [f32; 64] = [0u8; 64]
        .iter()
        .enumerate()
        .map(|(i, _v)| ((input >> i) & 1) as f32)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    out
}

impl<B: Backend> Batcher<DataItem, Batch<B>> for InternalBatcher<B> {
    fn batch(&self, items: Vec<DataItem>) -> Batch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();
        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(u64_to_bits(item.address), &self.device);
            inputs.push(input_tensor.unsqueeze());
        }
        let inputs = Tensor::cat(inputs, 0);

        let targets = items
            .iter()
            .map(|v| Tensor::<B, 1>::from_floats([v.value as f32], &self.device))
            .collect::<Vec<Tensor<B, 1>>>();
        let targets = Tensor::cat(targets, 0).to_device(&self.device);
        Batch {
            addresses: inputs,
            targets,
        }
    }
}
