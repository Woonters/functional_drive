use burn::tensor::{backend::Backend, Tensor};
use nbdkit::*;

#[derive(Default)]
struct MyDrive {
    // The drive code I think??
    _not_used: i32,
}

// temp learn burn

fn computation<B: Backend>() {
    let device = Default::default();
    let tensor1: Tensor<B, 2> = Tensor::from_floats([[2., 3.], [2., 5.]], &device);
    let tensor2 = Tensor::ones_like(&tensor1);
    println!("{:}", tensor1 + tensor2);
}

pub fn learn() {
    computation::<burn::backend::Wgpu>();
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
        (0..buf.len()).for_each(|byte| {
            buf[byte] = y_equals_zero(offset + byte as u64);
        });
        todo!()
    }

    fn get_size(&self) -> Result<i64> {
        Ok(i64::MAX)
    }
}

// let's just define a few math functions to develop with, I'm learning the nbdkit api to start with

fn y_equals_zero(_x: u64) -> u8 {
    0
}

plugin!(MyDrive {});
