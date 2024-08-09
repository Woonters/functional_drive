#[allow(unused_variables)]
use burn::{
    backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu},
    config::Config,
    data::dataloader::batcher::Batcher,
};
mod nn_backend;
use nbdkit::*;
use nn_backend::*;
use trainer::TrainingConfig;

#[derive()]
struct MyDrive {
    device: burn::backend::wgpu::WgpuDevice,
    trainer_conf: TrainingConfig,
    model: model::Model<Wgpu<AutoGraphicsApi, f32, i32>>,
}

impl Default for MyDrive {
    fn default() -> Self {
        type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
        type MyAutoDiffBackend = Autodiff<MyBackend>;
        let device = burn::backend::wgpu::WgpuDevice::default();
        let trainer_conf = TrainingConfig::load("/temp/guide/config.json")
            .expect("Config should exist for the model");
        Self {
            model: model::ModelConfig {
                input_size: 64,
                output_size: 1,
            }
            .init(&device),
            trainer_conf,
            device,
        }
    }
}

impl Server for MyDrive {
    fn name() -> &'static str {
        "The-worlds-first-functional-drive"
    }
    fn open(_readonly: bool) -> Result<Box<dyn Server>> {
        debug!("booting the drive | readonly={}", _readonly);
        Ok(Box::<MyDrive>::default())
    }

    fn read_at(&self, buf: &mut [u8], offset: u64) -> Result<()> {
        // let's calculate the bits we need to get: offset * 8
        let batcher = batcher::InternalBatcher::new(self.device.clone());
        let batch = batcher.batch(
            buf.iter()
                .enumerate()
                .map(|(i, &_v)| dataloader::DataItem {
                    address: i as u64 + offset,
                    value: 0u8,
                })
                .collect(),
        );
        self.model
            .forward(batch.addresses)
            .into_data()
            .value
            .iter()
            .enumerate()
            .for_each(|(i, &v)| buf[i] = v as u8);
        Ok(())
    }
    #[allow(unused_variables)]
    fn write_at(&self, buf: &[u8], offset: u64, flags: Flags) -> Result<()> {
        todo!();
        // self.nn_train(burn::backend::wgpu::WgpuDevice::default())
    }

    fn get_size(&self) -> Result<i64> {
        Ok(i64::MAX)
    }
}

impl MyDrive {
    // fn nn_train(&self, device: B::Device) -> Result<()> {
    //     type B = Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>;
    //     B::seed(self.trainer_conf.seed);
    //     let batcher_train = batcher::InternalBatcher::<B>::new(device.clone());
    //     let batcher_valid = batcher::InternalBatcher::<B::InnerBackend>::new(device.clone());
    //     let dataloader_train = DataLoaderBuilder::new(batcher_train)
    //         .batch_size(self.trainer_conf.batch_size)
    //         .shuffle(self.trainer_conf.seed)
    //         .num_workers(self.trainer_conf.num_workers)
    //         .build(dataloader::CustomDataset::new());
    //     let dataloader_test = DataLoaderBuilder::new(batcher_valid)
    //         .batch_size(self.trainer_conf.batch_size)
    //         .shuffle(self.trainer_conf.seed)
    //         .num_workers(self.trainer_conf.num_workers)
    //         .build(dataloader::CustomDataset::new());
    //     // for epoch in 1..self.trainer_conf.num_epochs + 1 {
    //     //     for (iteration, batch) in dataloader_train.iter().enumerate() {
    //     //         let output = self.model.forward(batch.addresses).into_data();
    //     //     }
    //     // }
    //     Ok(())
    // }
}

plugin!(MyDrive {});
