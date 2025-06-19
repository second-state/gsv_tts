use std::{collections::HashMap, sync::Arc};

use axum::body;
use axum::{body::Body, http::Response, response::IntoResponse};
use futures::StreamExt;
use gpt_sovits_rs::gsv::{SSL, SpeakerV2Pro, T2S, Vits};
use gpt_sovits_rs::tch::{Device, Tensor};
use gpt_sovits_rs::text::G2p;

pub mod config {
    #[derive(Debug, Clone, serde::Deserialize)]
    pub struct TTSConfig {
        #[serde(default)]
        pub bert_model_path: Option<String>,

        #[serde(default)]
        pub g2pw_model_path: Option<String>,

        pub ssl_model_path: String,
        pub mini_bart_g2p_path: String,

        pub speaker: Vec<SpeakerTTSConfig>,

        #[serde(default)]
        pub buffer_size: usize,
    }

    #[derive(Debug, Clone, serde::Deserialize)]
    pub struct SpeakerTTSConfig {
        pub name: String,
        pub t2s_path: String,
        pub vits_path: String,
        pub ref_audio_path: String,
        pub ref_text: String,
    }
}

#[derive(Debug, serde::Deserialize)]
pub struct TTSRequest {
    pub input: String,
    #[serde(alias = "voice")]
    pub speaker: String,
    #[serde(default)]
    pub response_format: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum TTSType {
    Full,
    Batch,
    Stream,
}

type TTSRespTx = tokio::sync::mpsc::UnboundedSender<Vec<f32>>;

pub struct SpeakerBuilder {
    vits: HashMap<String, Arc<Vits>>,
    t2s: HashMap<String, Arc<T2S>>,
    ssl: Arc<SSL>,
    device: Device,
}

impl SpeakerBuilder {
    pub fn new(ssl_path: &str) -> anyhow::Result<Self> {
        let device = Device::cuda_if_available();
        assert!(device.is_cuda(), "Currently only cuda is supported");
        let ssl = Arc::new(SSL::new(ssl_path, device)?);
        Ok(Self {
            vits: HashMap::new(),
            t2s: HashMap::new(),
            ssl,
            device,
        })
    }

    pub fn create_speaker(
        &mut self,
        name: &str,
        t2s_path: &str,
        vits_path: &str,
    ) -> anyhow::Result<SpeakerV2Pro> {
        let maybe_vits = self.vits.get(t2s_path).map(Arc::clone);
        let vits = if let Some(vits) = maybe_vits {
            vits
        } else {
            let vits = Arc::new(Vits::new(vits_path, self.device)?);
            self.vits.insert(t2s_path.to_string(), vits.clone());
            vits
        };

        let maybe_t2s = self.t2s.get(t2s_path).map(Arc::clone);
        let t2s = if let Some(t2s) = maybe_t2s {
            t2s
        } else {
            let t2s = Arc::new(T2S::new(t2s_path, self.device)?);
            self.t2s.insert(t2s_path.to_string(), t2s.clone());
            t2s
        };

        let ssl = self.ssl.clone();

        Ok(SpeakerV2Pro::new(name, t2s, vits, ssl))
    }
}

struct TTSSpeaker {
    speaker: SpeakerV2Pro,
    ///    (prompts, refer, sv_emb)
    refer: (Tensor, Tensor, Tensor),
    ref_phone: Tensor,
    ref_bert: Tensor,
}

pub struct TTSEngine {
    pub speakers: HashMap<String, TTSSpeaker>,
    pub g2p: G2p,
}

impl TTSEngine {
    pub fn new(g2p: G2p) -> Self {
        Self {
            speakers: HashMap::new(),
            g2p,
        }
    }

    pub fn add_speaker(
        &mut self,
        speaker: SpeakerV2Pro,
        ref_audio32k_samples: &[f32],
        device: Device,
        ref_text: &str,
    ) -> anyhow::Result<()> {
        let (ref_phone, ref_bert) = gpt_sovits_rs::text::get_phone_and_bert(&self.g2p, ref_text)?;
        let ref_bert = ref_bert.internal_cast_half(false);
        let ref_audio_32k = Tensor::from_slice(&ref_audio32k_samples)
            .internal_cast_half(false)
            .to_device(device)
            .unsqueeze(0);
        let refer = speaker.pre_handle_ref(ref_audio_32k)?;
        self.speakers.insert(
            speaker.name.clone(),
            TTSSpeaker {
                speaker,
                refer,
                ref_phone,
                ref_bert,
            },
        );

        Ok(())
    }

    pub fn list_speaker(&self) -> Vec<String> {
        self.speakers.keys().cloned().collect()
    }

    pub fn infer(&self, name: &str, text: &str) -> anyhow::Result<Vec<f32>> {
        let _g = gpt_sovits_rs::tch::no_grad_guard();
        let speaker = self
            .speakers
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("speaker {} not found", name,))?;

        let mut audios = vec![];

        for text in gpt_sovits_rs::text::split_text(text, 100) {
            let (text_phone, text_bert) = gpt_sovits_rs::text::get_phone_and_bert(&self.g2p, text)?;
            let text_bert = text_bert.internal_cast_half(false);

            let audio_32k = speaker.speaker.infer(
                (
                    speaker.refer.0.shallow_clone(),
                    speaker.refer.1.shallow_clone(),
                    speaker.refer.2.shallow_clone(),
                ),
                speaker.ref_phone.shallow_clone(),
                text_phone,
                speaker.ref_bert.shallow_clone(),
                text_bert,
                15,
            );
            match audio_32k {
                Ok(audio) => {
                    let audio_size = audio.size1().unwrap() as usize;
                    if audio_size == 0 {
                        log::warn!("infer {text} got empty audio, skip!");
                        continue;
                    }

                    audios.push(audio);
                }
                Err(e) => {
                    log::debug!("infer {text} failed: {}", e);
                    log::warn!("infer {text} failed, skip!");
                }
            }
        }

        if audios.is_empty() {
            return Ok(Vec::new());
        }

        let audio = Tensor::cat(&audios, 0);
        let audio_size = audio.size1().unwrap() as usize;

        let mut samples = vec![0f32; audio_size];
        audio
            .f_copy_data(&mut samples, audio_size)
            .map_err(|e| anyhow::anyhow!("copy data failed: {}", e))?;

        Ok(samples)
    }

    pub fn batch_infer<F: FnMut(Vec<f32>) -> anyhow::Result<()>>(
        &self,
        name: &str,
        text: &str,
        mut callback: F,
    ) -> anyhow::Result<()> {
        let _g = gpt_sovits_rs::tch::no_grad_guard();

        let speaker = self
            .speakers
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("speaker {} not found", name,))?;

        for text in gpt_sovits_rs::text::split_text(text, 100) {
            let (text_phone, text_bert) = gpt_sovits_rs::text::get_phone_and_bert(&self.g2p, text)?;
            let text_bert = text_bert.internal_cast_half(false);

            let audio_32k = speaker.speaker.infer(
                (
                    speaker.refer.0.shallow_clone(),
                    speaker.refer.1.shallow_clone(),
                    speaker.refer.2.shallow_clone(),
                ),
                speaker.ref_phone.shallow_clone(),
                text_phone,
                speaker.ref_bert.shallow_clone(),
                text_bert,
                15,
            );
            match audio_32k {
                Ok(audio) => {
                    let audio_size = audio.size1().unwrap() as usize;
                    if audio_size == 0 {
                        log::warn!("infer {text} got empty audio, skip!");
                        continue;
                    }

                    let mut samples = vec![0f32; audio_size];
                    audio
                        .f_copy_data(&mut samples, audio_size)
                        .map_err(|e| anyhow::anyhow!("copy data failed: {}", e))?;
                    callback(samples)?;
                }
                Err(e) => {
                    log::debug!("infer {text} failed: {}", e);
                    log::warn!("infer {text} failed, skip!");
                }
            }
        }

        Ok(())
    }

    pub fn stream_infer<F: FnMut(Vec<f32>) -> anyhow::Result<()>>(
        &self,
        name: &str,
        text: &str,
        mut callback: F,
    ) -> anyhow::Result<()> {
        let _g = gpt_sovits_rs::tch::no_grad_guard();

        let speaker = self
            .speakers
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("speaker {} not found", name,))?;

        for text in gpt_sovits_rs::text::split_text(text, 100) {
            let (text_phone, text_bert) = gpt_sovits_rs::text::get_phone_and_bert(&self.g2p, text)?;
            let text_bert = text_bert.internal_cast_half(false);

            let mut s = speaker.speaker.stream_infer(
                (
                    speaker.refer.0.shallow_clone(),
                    speaker.refer.1.shallow_clone(),
                    speaker.refer.2.shallow_clone(),
                ),
                speaker.ref_phone.shallow_clone(),
                text_phone,
                speaker.ref_bert.shallow_clone(),
                text_bert,
                15,
            )?;
            while let Some(chunk) = s.next_chunk(30, &[25, 25, 50, 75])? {
                let audio_size = chunk.size1().unwrap() as usize;
                if audio_size == 0 {
                    log::warn!("infer {text} got empty audio, skip!");
                    continue;
                }

                let mut samples = vec![0f32; audio_size];
                chunk
                    .f_copy_data(&mut samples, audio_size)
                    .map_err(|e| anyhow::anyhow!("copy data failed: {}", e))?;
                callback(samples)?;
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TTSService {
    pub tx: tokio::sync::mpsc::Sender<(TTSRequest, TTSType, TTSRespTx)>,
    pub speaker_name_list: Arc<Vec<String>>,
}

fn load_32k_wav(path: &str) -> anyhow::Result<Vec<f32>> {
    let fs = std::fs::File::open(path)?;
    let (header, mut audio_samples) =
        wav_io::read_from_file(fs).map_err(|e| anyhow::anyhow!("read wav file failed: {}", e))?;

    if header.channels != 1 {
        let (sample, _) = wav_io::utils::split_stereo_wave(audio_samples);
        audio_samples = sample;
    }

    if header.sample_rate != 32000 {
        log::info!("wav sample rate: {}, need 32000", header.sample_rate);
        audio_samples = wav_io::resample::linear(audio_samples, 1, header.sample_rate, 32000);
    }

    Ok(audio_samples)
}

impl TTSService {
    pub fn create_with_config(config: config::TTSConfig) -> anyhow::Result<Self> {
        let _g = gpt_sovits_rs::tch::no_grad_guard();

        let buffer_size = if config.buffer_size == 0 {
            5
        } else {
            config.buffer_size
        };

        let device = Device::cuda_if_available();
        assert!(device.is_cuda(), "Currently only cuda is supported");

        let mut builder = SpeakerBuilder::new(&config.ssl_model_path)?;
        let mut g2p_conf = gpt_sovits_rs::text::G2PConfig::new(config.mini_bart_g2p_path);
        if config.g2pw_model_path.is_some() && config.bert_model_path.is_some() {
            g2p_conf = g2p_conf.with_chinese(
                config.g2pw_model_path.unwrap(),
                config.bert_model_path.unwrap(),
            )
        }

        let g2p = g2p_conf.build(device)?;
        let mut engine = TTSEngine::new(g2p);

        for speaker in config.speaker {
            let audio_32k = load_32k_wav(&speaker.ref_audio_path)?;
            let s = builder.create_speaker(&speaker.name, &speaker.t2s_path, &speaker.vits_path)?;
            engine.add_speaker(s, &audio_32k, device, &speaker.ref_text)?;
        }

        Ok(Self::new(engine, buffer_size))
    }

    pub fn new(engine: TTSEngine, buffer: usize) -> Self {
        let speaker_name_list = Arc::new(engine.list_speaker());

        let (tx, rx) = tokio::sync::mpsc::channel(buffer);
        tokio::task::spawn_blocking(move || {
            Self::tts_loop(engine, rx);
        });
        Self {
            tx,
            speaker_name_list,
        }
    }

    fn tts_loop(
        engine: TTSEngine,
        mut rx: tokio::sync::mpsc::Receiver<(TTSRequest, TTSType, TTSRespTx)>,
    ) {
        while let Some((req, tts_type, tx)) = rx.blocking_recv() {
            if tx.is_closed() {
                continue;
            }
            if req.input.is_empty() {
                log::warn!("tts request input is empty, skip!");
                continue;
            }
            log::info!("tts request: {:?}", req);
            if !engine.list_speaker().contains(&req.speaker) {
                log::warn!("speaker {} not found, skip!", req.speaker);
                continue;
            }
            match tts_type {
                TTSType::Full => match engine.infer(&req.speaker, &req.input) {
                    Ok(audio) => {
                        if tx.send(audio).is_err() {
                            log::warn!("tts response channel closed, skip!");
                        }
                    }
                    Err(e) => {
                        log::error!("tts error: {}", e);
                    }
                },
                TTSType::Batch => {
                    if let Err(e) = engine.batch_infer(&req.speaker, &req.input, |audio| {
                        tx.send(audio)
                            .map_err(|_| anyhow::anyhow!("send audio to request_tx failed"))
                    }) {
                        log::error!("{req:?} tts error: {}", e);
                        continue;
                    }
                }
                TTSType::Stream => {
                    if let Err(e) = engine.stream_infer(&req.speaker, &req.input, |audio| {
                        tx.send(audio)
                            .map_err(|_| anyhow::anyhow!("send audio to request_tx failed"))
                    }) {
                        log::error!("{req:?} tts error: {}", e);
                        continue;
                    }
                }
            }
        }
        log::warn!("tts_loop exit");
    }
}

pub async fn tts_service(
    tts_service: axum::extract::Extension<TTSService>,
    req: axum::Json<TTSRequest>,
) -> impl IntoResponse {
    log::info!("tts request: {:?}", req);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let e = tts_service.tx.send((req.0, TTSType::Full, tx)).await;
    if e.is_err() {
        return Response::builder()
            .status(500)
            .body(Body::from("TTS service is not available"))
            .unwrap();
    };

    match rx.recv().await {
        Some(s) => {
            let header = wav_io::new_header(32000, 16, false, true);
            let body = wav_io::write_to_bytes(&header, &s);
            match body {
                Ok(body) => Response::builder()
                    .header(axum::http::header::CONTENT_TYPE, "audio/wav")
                    .body(Body::from(body))
                    .unwrap(),
                Err(e) => {
                    log::error!("write wav to bytes failed: {}", e);
                    Response::builder()
                        .status(500)
                        .body(Body::from("Internal Server Error"))
                        .unwrap()
                }
            }
        }
        None => {
            todo!()
        }
    }
}

pub async fn tts_batch_service(
    tts_service: axum::extract::Extension<TTSService>,
    req: axum::Json<TTSRequest>,
) -> impl IntoResponse {
    log::info!("tts_batch_service request: {:?}", req);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let e = tts_service.tx.send((req.0, TTSType::Batch, tx)).await;
    if e.is_err() {
        return Response::builder()
            .status(500)
            .body(Body::from("TTS service is not available"))
            .unwrap();
    };

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|samples| {
        let mut samples_i16 = vec![];
        for v in samples {
            let v = (v * std::i16::MAX as f32) as i16;
            let v = i16::to_le_bytes(v);
            samples_i16.push(v[0]);
            samples_i16.push(v[1]);
        }
        let r: Result<Vec<u8>, &'static str> = Ok(samples_i16);
        r
    });

    let audio_stream = axum::body::Body::from_stream(stream);
    Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "audio/pcm;rate=32000")
        .body(audio_stream)
        .unwrap()
}

pub async fn tts_stream_service(
    tts_service: axum::extract::Extension<TTSService>,
    req: axum::Json<TTSRequest>,
) -> impl IntoResponse {
    log::info!("tts stream request: {:?}", req);

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    let e = tts_service.tx.send((req.0, TTSType::Stream, tx)).await;
    if e.is_err() {
        return Response::builder()
            .status(500)
            .body(Body::from("TTS service is not available"))
            .unwrap();
    };

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|samples| {
        let mut samples_i16 = vec![];
        for v in samples {
            let v = (v * std::i16::MAX as f32) as i16;
            let v = i16::to_le_bytes(v);
            samples_i16.push(v[0]);
            samples_i16.push(v[1]);
        }
        log::debug!("send chunk: {}", samples_i16.len());
        let r: Result<Vec<u8>, &'static str> = Ok(samples_i16);
        r
    });

    let audio_stream = axum::body::Body::from_stream(stream);
    Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "audio/pcm;rate=32000")
        .body(audio_stream)
        .unwrap()
}

pub async fn tts_speakers_service(
    tts_service: axum::extract::Extension<TTSService>,
) -> impl IntoResponse {
    Response::builder()
        .header(axum::http::header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_string(tts_service.speaker_name_list.as_ref()).unwrap(),
        ))
        .unwrap()
}
