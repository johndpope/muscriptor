use std::io::Read;
use std::path::{Path, PathBuf};
use std::fs;

pub fn download_weights(url: &str) -> Result<PathBuf, DownloadError> {
    let cache_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".cache")
        .join("muscriptor");
    fs::create_dir_all(&cache_dir).map_err(DownloadError::Io)?;

    let path_in_repo = url
        .strip_prefix("hf://")
        .ok_or_else(|| DownloadError::Format("URL must start with hf://".to_string()))?;

    let parts: Vec<&str> = path_in_repo.splitn(3, '/').collect();
    if parts.len() != 3 {
        return Err(DownloadError::Format(format!("Invalid hf:// URL: {}", url)));
    }
    let org = parts[0];
    let name = parts[1];
    let filename = parts[2];
    let repo_id = format!("{}/{}", org, name);
    let api_url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);

    let cached_path = cache_dir.join(format!("{}_{}", name, filename));
    if cached_path.exists() {
        log::info!("Using cached weights at {}", cached_path.display());
        return Ok(cached_path);
    }

    log::info!("Downloading {} from {} ...", filename, api_url);

    let token = std::env::var("HF_TOKEN").ok();
    let resp = if let Some(t) = &token {
        ureq::get(&api_url)
            .set("Authorization", &format!("Bearer {}", t))
            .call()
            .map_err(|e| DownloadError::Http(e.to_string()))?
    } else {
        ureq::get(&api_url)
            .call()
            .map_err(|e| DownloadError::Http(e.to_string()))?
    };

    let mut reader = resp.into_reader();
    let mut body: Vec<u8> = Vec::new();
    reader.read_to_end(&mut body).map_err(|e| DownloadError::Http(e.to_string()))?;

    if body.len() < 100 && body.starts_with(b"<") {
        return Err(DownloadError::Auth(format!(
            "Cannot download '{}': the MuScriptor model weights are gated. \
             Set HF_TOKEN or run `huggingface-cli login`.",
            repo_id
        )));
    }

    fs::write(&cached_path, &body).map_err(DownloadError::Io)?;
    log::info!("Downloaded {} bytes to {}", body.len(), cached_path.display());

    // Also try config.json
    let config_url = format!("https://huggingface.co/{}/resolve/main/config.json", repo_id);
    let config_path = cache_dir.join(format!("{}_config.json", name));
    if !config_path.exists() {
        if let Ok(cfg_resp) = ureq::get(&config_url).call() {
            let mut rdr = cfg_resp.into_reader();
            let mut cfg_body = String::new();
            if rdr.read_to_string(&mut cfg_body).is_ok() && !cfg_body.starts_with('<') {
                let _ = fs::write(&config_path, &cfg_body);
            }
        }
    }

    Ok(cached_path)
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadError {
    #[error("Invalid URL format: {0}")]
    Format(String),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("Authentication error: {0}")]
    Auth(String),
    #[error("IO error: {0}")]
    Io(std::io::Error),
}
