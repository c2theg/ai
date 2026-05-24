#!/bin/bash
#  Updated: 5/24/2026
#  Version: 0.0.31
#  Purpose:  Downloads a list of LLM Models into Ollama hosted locally in a docker container
#  Install:
#       wget -O "install_ai_models_ollama_v128Gb.sh" https://raw.githubusercontent.com/c2theg/srvBuilds/refs/heads/master/install_ai_models_ollama_v128Gb.sh && chmod +x install_ai_models_ollama_v128Gb.sh
#
#-----------------------------------------------------------------------------------------
# Create the directory (Just incase its not present)
sudo mkdir -p /usr/share/ollama/models
# Give the directory appropriate permissions for the Docker container
sudo chmod -R 777 /usr/share/ollama/models

cd /usr/share/ollama/models

echo "Checking if Ollama is ready..."
until $(curl --output /dev/null --silent --head --fail http://localhost:11434); do
    printf '.'
    sleep 2
done

echo -e "\nOllama is up! \n\n\n"


echo -e "\nUpdate all models\n"
ollama list | tail -n +2 | awk '{print $1}' | xargs -I {} ollama pull {}
echo -e "\nAll models updated\n"


#-----------------------------------------------------------------------------------------
# AI Models
#-----------------------------------------------------------------------------------------


echo "

Downloading AI Models...

"

# Under 16Gb vRam
# "llama3.2:latest" "minimax-m2.1:cloud" "qwen3-embedding:0.6b" "ministral-3:8b" "qwen3-vl:8b

# 128Gb vRam
# qwen3.6:27b-q8_0
MODELS=("qwen3.6:27b-q4_K_M" "qwen3-embedding:4b" "nomic-embed-text-v2-moe:latest")

for MODEL in "${MODELS[@]}"; do
    echo "
    ------------------------------------------
    "
    echo "Downloading $MODEL..."
    #docker exec -it ollama ollama pull $MODEL
    ollama pull $MODEL
done

echo "

------------------------------------------

"

echo "

Downloading Enhanced AIModels...

"
#  "gemma3:4b"

MODELS=("qwen2.5:7b" "qwen2.5vl:7b" "qwen3:14b-q4_K_M" "mistral-small3.2:24b-instruct-2506-q4_K_M" "llama3.2:3b-instruct-q8_0"  "qwen3-embedding:4b" "granite3.2-vision")

for MODEL in "${MODELS[@]}"; do
    echo "
    ------------------------------------------
    "
    echo "Downloading $MODEL..."
    #docker exec -it ollama ollama pull $MODEL
    ollama pull $MODEL
done

echo "

------------------------------------------


Setup complete! Models are stored in /usr/share/ollama/models

All models installed successfully!

You can now select them in Open WebUI at http://localhost:3000


"

#--- install other ML / NLP Models ----

mkdir -p /opt/python3_shared
cd /opt/python3_shared


#/media/data/sync/ai_personal/code/ai_image_to_text/models/openai--clip-vit-large-patch14/pytorch_model.bin
#/media/data/sync/ai_personal/code/ai_image_to_text/models/openai--clip-vit-large-patch14/model.safetensors
#/media/data/sync/ai_personal/code/ai_image_to_text/models/openai--clip-vit-large-patch14/_torch_model.pt


#----
# /opt/ai_shared/models/ggml-large-v3.bin
# /opt/ai_shared/models/ggml-large-v3-turbo.bin
# /opt/ai_shared/models/ggml-large-v3-turbo-q8_0.bin
# /opt/ai_shared/models/ggml-medium.en-q8_0.bin
# /opt/ai_shared/whisper.cpp/models/ggml-medium.en.bin



# delete - cuda-repo-ubuntu2404-13-1-local_13.1.0-590.44.01-1_amd64.deb


#--- images -----
# yolo11n.pt
# buffalo_l
# openai/clip-vit-large-patch14
