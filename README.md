# An API server for streaming TTS

## Install libtorch dependencies

Run the following commands under Linux.

```
# download libtorch
curl -LO https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.0%2Bcu124.zip

unzip libtorch-cxx11-abi-shared-with-deps-2.4.0+cu124.zip

# Add to ~/.zprofile or ~/.bash_profile
export LD_LIBRARY_PATH=$(pwd)/libtorch/lib:$LD_LIBRARY_PATH
export LIBTORCH=$(pwd)/libtorch 
```

## Build the API server

```
git clone https://github.com/second-state/gsv_tts
git clone https://github.com/second-state/gpt_sovits_rs

cd gsv_tts
cargo build --release
```

## Get model files

```
cd resources

curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/t2s.pt
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/vits.pt
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/ssl_model.pt
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/bert_model.pt
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/g2pw_model.pt
curl -LO https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/mini-bart-g2p.pt
```

If you do not have Nvidia GPU and CUDA installed. You will download the following CPU over-write versions. 

```
curl -L -o t2s.pt https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/t2s.cpu.pt
curl -L -o vits.pt https://huggingface.co/L-jasmine/GPT_Sovits/resolve/main/v2pro/vits.cpu.pt
```

## Run the API server

```
TTS_LISTEN=0.0.0.0:9094 nohup target/release/gsv_tts &
```

Test the server here: http://localhost:9094/

