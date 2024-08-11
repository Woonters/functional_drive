use core::fmt;
use std::cell::RefCell;

use burn::{
    data::{
        dataloader::{batcher::Batcher, DataLoaderBuilder},
        dataset::InMemDataset,
    },
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, ElementConversion},
    train::{metric::LossMetric, LearnerBuilder},
};

use crate::{
    batcher,
    dataloader::{self, CustomDataset, DataItem},
    model::ModelConfig,
    trainer::TrainingConfig,
};

#[derive(Debug, Clone)]
pub struct NNError;

impl fmt::Display for NNError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Something went wrong with training the NN") // TODO: This error should be a lot better
    }
}

pub struct TheNetwork<A: AutodiffBackend> {
    // This will be the actual network along with all the associated functions for handling training the new network and getting info from it (infering / reading)
    model: RefCell<super::model::Model<A>>,
    training_config: TrainingConfig,
    device: A::Device,
    max_size: usize,
}

impl<A: AutodiffBackend> TheNetwork<A> {
    pub fn init() -> Self {
        let device = A::Device::default();
        let model_config = ModelConfig::new(64, 1);
        let model = model_config.init::<A>(&device);
        let training_config = TrainingConfig::new(model_config, AdamConfig::new());
        Self {
            model: RefCell::new(model),
            training_config,
            device,
            max_size: 0,
        }
    }

    pub fn read_at(&self, buf: &mut [u8], offset: u64) -> Result<(), NNError> {
        // let's calculate the bits we need to get: offset * 8
        let batcher = super::batcher::InternalBatcher::<A>::new(self.device.clone());
        let batch = batcher.batch(
            buf.iter()
                .enumerate()
                .map(|(i, &_v)| super::dataloader::DataItem {
                    address: i as u64 + offset,
                    value: 0u8,
                })
                .collect(),
        );
        let out: Vec<_> = self
            .model
            .borrow()
            .forward(batch.addresses)
            .into_data()
            .value;
        out.iter()
            .enumerate()
            .for_each(|(i, v)| buf[i] = v.elem::<f32>() as u8);
        Ok(())
    }

    pub fn train(&self, buf: &[u8], offset: u64) -> Result<(), NNError> {
        A::seed(self.training_config.seed);
        let batcher_train = batcher::InternalBatcher::<A>::new(self.device.clone());
        let batcher_valid = batcher::InternalBatcher::<A::InnerBackend>::new(self.device.clone()); // TODO: Got to work out this line here not sure what I can really do about it though
        let training_dataset = dataloader::CustomDataset::retrain(
            CustomDataset {
                overwrite_buf: Some(buf.to_vec()),
                buf_len: Some(buf.len()),
                overwrite_offset: Some(offset as usize),
                dataset: InMemDataset::new(vec![DataItem {
                    address: 0,
                    value: 0,
                }]),
                max_size: Some(self.max_size),
            },
            &self.device,
            self.model.borrow().clone(),
        );
        let testing_dataset = dataloader::CustomDataset::retrain(
            CustomDataset {
                overwrite_buf: Some(buf.to_vec()),
                buf_len: Some(buf.len()),
                overwrite_offset: Some(offset as usize),
                dataset: InMemDataset::new(vec![DataItem {
                    address: 0,
                    value: 0,
                }]),
                max_size: Some(self.max_size),
            },
            &self.device,
            self.model.borrow().clone(),
        );
        let dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(self.training_config.batch_size)
            .shuffle(self.training_config.seed)
            .num_workers(self.training_config.num_workers)
            .build(training_dataset);
        let dataloader_test = DataLoaderBuilder::new(batcher_valid)
            .batch_size(self.training_config.batch_size)
            .shuffle(self.training_config.seed)
            .num_workers(self.training_config.num_workers)
            .build(testing_dataset);
        let learner = LearnerBuilder::new("/tmp/guide")
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .with_file_checkpointer(CompactRecorder::new())
            .devices(vec![self.device.clone()])
            .num_epochs(self.training_config.num_epochs)
            .build(
                self.model.borrow().clone(),
                self.training_config.optimizer.init(),
                self.training_config.learning_rate,
            );
        let model_trained = learner.fit(dataloader_train, dataloader_test);
        let mut v = self.model.borrow_mut();
        *v = model_trained;
        todo!()
    }
}
