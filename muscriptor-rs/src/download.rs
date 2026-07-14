use std::io::Read;
use std::path::PathBuf;
use std::fs;

/// Resolve a HuggingFace token: the `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` env
/// vars first, then the standard token file written by `hf auth login`
/// (`$HF_HOME/token` or `~/.cache/huggingface/token`).
fn hf_token() -> Option<String> {
    for var in ["HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(t) = std::env::var(var) {
            let t = t.trim().to_string();
            if !t.is_empty() {
                return Some(t);
            }
        }
    }
    let token_path = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".cache")
                .join("huggingface")
        })
        .join("token");
    fs::read_to_string(token_path).ok().map(|t| t.trim().to_string()).filter(|t| !t.is_empty())
}

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

    let token = hf_token();
    let make_req = || {
        let req = ureq::get(&api_url);
        if let Some(t) = &token {
            req.set("Authorization", &format!("Bearer {}", t))
        } else {
            req
        }
    };
    let resp = match make_req().call() {
        Ok(r) => r,
        Err(ureq::Error::Status(code, _)) => {
            // 401 = no/invalid token was sent; 403 = authenticated but the
            // gated repo's terms haven't been accepted for this account.
            if code == 401 || code == 403 {
                let had_token = token.is_some();
                let cause = if code == 401 && !had_token {
                    "no HuggingFace token was found"
                } else if code == 401 {
                    "the HuggingFace token was rejected (invalid or expired)"
                } else {
                    "your account hasn't been granted access to this gated model"
                };
                return Err(DownloadError::Auth(format!(
                    "Cannot download '{repo_id}' (HTTP {code}): {cause}.\n\
                     1. Accept the license at https://huggingface.co/{repo_id}\n\
                     2. Provide a token (create one at https://huggingface.co/settings/tokens), either:\n\
                     \x20     export HUGGINGFACE_TOKEN=hf_...   (also accepts HF_TOKEN / HUGGING_FACE_HUB_TOKEN)\n\
                     \x20  or run `hf auth login` (writes ~/.cache/huggingface/token, which this tool reads).\n\
                     (Token source: {}.)",
                    if had_token { "a token was found but the server rejected it" } else { "none found in env vars or ~/.cache/huggingface/token" }
                )));
            }
            return Err(DownloadError::Http(format!("status code {}", code)));
        }
        Err(e) => return Err(DownloadError::Http(e.to_string())),
    };

    let mut reader = resp.into_reader();
    let mut body: Vec<u8> = Vec::new();
    reader.read_to_end(&mut body).map_err(|e| DownloadError::Http(e.to_string()))?;

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
        } else {
            // config download failure is non-fatal
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
