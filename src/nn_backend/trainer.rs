use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::Autodiff;
use burn::backend::Wgpu;
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::train::metric::LossMetric;
use burn::train::LearnerBuilder;

fn main() {
    println!("Hello World!");
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutoDiffBackend = Autodiff<MyBackend>;
    let device = burn::backend::wgpu::WgpuDevice::default();
    train::<MyAutoDiffBackend>(
        "/tmp/guide",
        TrainingConfig::new(
            crate::model::ModelConfig {
                input_size: 64,
                output_size: 1,
            },
            AdamConfig::new(),
        ),
        device.clone(),
    );
    for i in 0..100 {
        infer::<MyBackend>(
            "/tmp/guide",
            device.clone(),
            super::dataloader::CustomDataset::new().get(i).unwrap(),
        );
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: crate::model::ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 20)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-2)]
    pub learning_rate: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: crate::model::ModelConfig {
                input_size: 64,
                output_size: 1,
            },
            optimizer: AdamConfig::new(),
            num_epochs: 20,
            batch_size: 64,
            num_workers: 4,
            seed: 42,
            learning_rate: 1.0e-2,
        }
    }
}

fn create_artifact_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");
    B::seed(config.seed);
    let batcher_train = super::batcher::InternalBatcher::<B>::new(device.clone());
    let batcher_valid = super::batcher::InternalBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(super::dataloader::CustomDataset::new());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(super::dataloader::CustomDataset::new());
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained Model should be saved");
}

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, item: super::dataloader::DataItem) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist");
    let model = config.model.init::<B>(&device).load_record(record);
    let label = item.value;
    let batcher = super::batcher::InternalBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.addresses);
    println!("Predicted {} | Expected {}", output.into_data(), label);
}
