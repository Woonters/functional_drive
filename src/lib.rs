use burn::backend::Autodiff;
#[allow(unused_variables)]
use burn::backend::{wgpu::AutoGraphicsApi, Wgpu};
mod nn_backend;
use interface::TheNetwork;
use nbdkit::*;
use nn_backend::*;
use trainer::TrainingConfig;

type NetworkClamped = TheNetwork<Autodiff<Wgpu<AutoGraphicsApi, f32, i32>>>;
#[derive()]
struct MyDrive {
    storage_network: NetworkClamped,
}

impl Default for MyDrive {
    fn default() -> Self {
        Self {
            storage_network: NetworkClamped::init(),
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
        match self.storage_network.read_at(buf, offset) {
            Ok(_) => Ok(()),
            Err(_) => Err(nbdkit::Error::new(0, "Something has gone wrong")),
        }
    }
    fn write_at(&self, buf: &[u8], offset: u64, _flags: Flags) -> Result<()> {
        match self.storage_network.train(buf, offset) {
            Ok(_) => Ok(()),
            Err(_) => Err(nbdkit::Error::new(0, "Something went wrong writing")),
        }
        // self.nn_train(burn::backend::wgpu::WgpuDevice::default())
    }

    fn get_size(&self) -> Result<i64> {
        Ok(i64::MAX)
    }
}

plugin!(MyDrive { write_at });
