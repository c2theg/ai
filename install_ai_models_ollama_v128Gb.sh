#!/bin/bash
#  Updated: 5/24/2026
#  Version: 0.0.34
#  Purpose:  Downloads a list of LLM Models into Ollama hosted locally in a docker container
#  Install:
#       wget -O "install_ai_models_ollama_v128Gb.sh" https://raw.githubusercontent.com/c2theg/ai/refs/heads/main/install_ai_models_ollama_v128Gb.sh && chmod +x install_ai_models_ollama_v128Gb.sh
#
#-----------------------------------------------------------------------------------------
# Create the directory (Just incase its not present)
#
# /opt/models
# /usr/share/ollama/models
#
#
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
# qwen3.6:27b-q8_0 "gpt-oss:20b"
# NVIDIA Nemotron 3 Nano Omni
MODELS=("qwen3-embedding:4b" "nomic-embed-text-v2-moe:latest" "qwen3.6:27b-q4_K_M" "nemotron3:33b-q4_K_M")

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

# echo "

# Downloading Enhanced AIModels...

# "

# #  "gemma3:4b"
# MODELS=("qwen2.5:7b" "qwen2.5vl:7b" "qwen3:14b-q4_K_M" "mistral-small3.2:24b-instruct-2506-q4_K_M" "llama3.2:3b-instruct-q8_0" "granite3.2-vision")
# for MODEL in "${MODELS[@]}"; do
#     echo "
#     ------------------------------------------
#     "
#     echo "Downloading $MODEL..."
#     #docker exec -it ollama ollama pull $MODEL
#     ollama pull $MODEL
# done

echo "

------------------------------------------


Setup complete! Models are stored in /usr/share/ollama/models

All models installed successfully!

You can now select them in Open WebUI at http://localhost:3000


"
