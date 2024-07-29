use burn::{
    module::Module,
    nn::{loss::MseLoss, Linear, LinearConfig, Relu},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use super::batcher::Batch;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    lin1: Linear<B>,
    lin2: Linear<B>,
    linrep1: Linear<B>,
    linrep2: Linear<B>,
    linrep3: Linear<B>,
    lin3: Linear<B>,
    lin4: Linear<B>,
    lin5: Linear<B>,
    activation: Relu,
}

#[derive(burn::config::Config, Debug, Default)]
pub struct ModelConfig {
    pub input_size: usize,
    pub output_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            lin1: LinearConfig::new(self.input_size, 32)
                .with_bias(true)
                .init(device),
            activation: Relu::new(),
            lin2: LinearConfig::new(32, 128).with_bias(true).init(device),
            linrep1: LinearConfig::new(128, 128).with_bias(true).init(device),
            linrep2: LinearConfig::new(128, 128).with_bias(true).init(device),
            linrep3: LinearConfig::new(128, 128).with_bias(true).init(device),
            lin4: LinearConfig::new(128, 512).with_bias(true).init(device),
            lin5: LinearConfig::new(512, 128).with_bias(true).init(device),
            lin3: LinearConfig::new(128, self.output_size)
                .with_bias(true)
                .init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.lin1.forward(x);
        let x = self.activation.forward(x);
        let x = self.lin2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linrep1.forward(x);
        let x = self.activation.forward(x);
        let x = self.linrep2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linrep3.forward(x);
        let x = self.activation.forward(x);
        let x = self.lin4.forward(x);
        let x = self.activation.forward(x);
        let x = self.lin5.forward(x);
        let x = self.activation.forward(x);
        let x = self.lin3.forward(x);
        self.activation.forward(x) // Would be better if it clamped values to (0,1)
    }

    pub fn forward_regression(&self, item: Batch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze();
        let output: Tensor<B, 2> = self.forward(item.addresses);
        let loss = MseLoss::new().forward(
            output.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        );
        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<super::batcher::Batch<B>, RegressionOutput<B>> for Model<B> {
    fn step(
        &self,
        item: super::batcher::Batch<B>,
    ) -> burn::train::TrainOutput<RegressionOutput<B>> {
        let item = self.forward_regression(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}
impl<B: Backend> ValidStep<super::batcher::Batch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: super::batcher::Batch<B>) -> RegressionOutput<B> {
        self.forward_regression(item)
    }
}
