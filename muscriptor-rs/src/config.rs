use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub card: usize,
    pub hidden_scale: usize,
    pub max_period: f64,
}

impl ModelConfig {
    pub fn dim_feedforward(&self) -> usize {
        self.hidden_scale * self.dim
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }
}

pub const DEFAULT_CONFIGS: &[(&str, ModelConfig)] = &[
    (
        "large",
        ModelConfig {
            dim: 1536,
            num_heads: 24,
            num_layers: 48,
            card: 1395,
            hidden_scale: 4,
            max_period: 10000.0,
        },
    ),
    (
        "medium",
        ModelConfig {
            dim: 1024,
            num_heads: 16,
            num_layers: 24,
            card: 1395,
            hidden_scale: 4,
            max_period: 10000.0,
        },
    ),
    (
        "small",
        ModelConfig {
            dim: 768,
            num_heads: 12,
            num_layers: 14,
            card: 1393,
            hidden_scale: 4,
            max_period: 10000.0,
        },
    ),
];
