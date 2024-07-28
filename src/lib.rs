use burn::{
    backend::{wgpu::AutoGraphicsApi, Wgpu},
    config::Config,
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
};
mod nn_backend;
use nbdkit::*;
use nn_backend::*;

#[derive(Default)]
struct MyDrive {
    device: burn::backend::wgpu::WgpuDevice,
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
        type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
        // let's calculate the bits we need to get: offset * 8
        let config = trainer::TrainingConfig::load(format!("/temp/guide/config.json"))
            .expect("Config should exist for the model");
        let record = CompactRecorder::new()
            .load(format!("/temp/guide/model").into(), &self.device)
            .expect("Trained model should exist");
        let model = config
            .model
            .init::<MyBackend>(&self.device)
            .load_record(record);
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
        model
            .forward(batch.addresses)
            .into_data()
            .value
            .iter()
            .enumerate()
            .for_each(|(i, &v)| buf[i] = v as u8);
        return Ok(());
    }

    fn get_size(&self) -> Result<i64> {
        Ok(i64::MAX)
    }
}

plugin!(MyDrive {});
