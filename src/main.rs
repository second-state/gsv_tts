use axum::{
    Extension, Router,
    response::Html,
    routing::{get, post},
};
use clap::{Parser, command};
use std::{collections::HashMap, fs};
use tts::config::TTSConfig;

use axum::extract::Query;

mod tts;

#[derive(Debug, Parser, Clone)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "127.0.0.1:8000", env("TTS_LISTEN"))]
    listen: String,
    #[arg(
        short,
        long,
        default_value = "config.json",
        env("GPT_SOVITS_CONFIG_PATH")
    )]
    config: String,
}

#[tokio::main]
async fn main() {
    env_logger::init();
    // build our application

    let args = Args::parse();

    log::info!("load config from {}", args.config);
    let config_data = std::fs::read_to_string(args.config).expect("read config file failed");
    let tts: TTSConfig = serde_json::from_str(&config_data).expect("parse config failed");

    let tts_state = tts::TTSService::create_with_config(tts).expect("create tts service failed");

    let app = app(tts_state);

    log::info!("listening on {}", args.listen);
    // run it
    let listener = tokio::net::TcpListener::bind(args.listen).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn app(tts_state: tts::TTSService) -> Router {
    // build our application with a route
    Router::new()
        .route("/v1/audio/speakers", get(tts::tts_speakers_service))
        .route("/v1/audio/stream_speech", post(tts::tts_stream_service))
        .route("/v1/audio/batch_speech", post(tts::tts_batch_service))
        .route("/v1/audio/speech", post(tts::tts_service))
        .fallback(get(index_page))
        .layer(Extension(tts_state))
}

#[cfg(debug_assertions)]
async fn index_page(Query(params): Query<HashMap<String, String>>) -> Html<String> {
    log::info!("Running in debug mode, reading index.html from resources directory");

    match params.get("lang").map(|s| s.as_str()) {
        Some("zh") => {
            let index_zh_html = fs::read_to_string("./resources/index.zh.html")
                .expect("Failed to read index.zh.html");
            Html(index_zh_html)
        }
        _ => {
            let index_html =
                fs::read_to_string("./resources/index.html").expect("Failed to read index.html");
            Html(index_html)
        }
    }
}

#[cfg(not(debug_assertions))]
async fn index_page(Query(params): Query<HashMap<String, String>>) -> Html<&'static str> {
    static INDEX_HTML: &str = include_str!("../resources/index.html");
    static INDEX_ZH_HTML: &str = include_str!("../resources/index.zh.html");
    match params.get("lang").map(|s| s.as_str()) {
        Some("zh") => Html(INDEX_ZH_HTML),
        _ => Html(&INDEX_HTML),
    }
}
