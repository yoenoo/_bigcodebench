rm -rf bcb_results/*

uv pip install bigcodebench --upgrade
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation

uv pip install matplotlib scikit-learn seaborn folium scikit-image nltk statsmodels bs4 faker flask openpyxl xmltodict pyquery cryptography
uv pip install django holidays geopy tensorflow keras sendgrid docx xlwt prettytable flask_mail natsort chardet pytesseract docx wordninja textblob python-Levenshtein
uv pip install mechanize texttable exceptions pycryptodome gensim pyfakefs
uv pip install wordcloud wikipedia flask_login flask_restful librosa requests_mock mechanize texttable python-docx pycryptodome gensim pyfakefs
uv pip install flask_wtf geopandas 

apt-get update && apt-get install -y python3-tk
apt-get update && apt-get install -y python3.10-dev build-essential

 
DATASET=bigcodebench
# MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
MODEL="Qwen/Qwen3-14B"
BACKEND=vllm
NUM_GPU=1
SPLIT=complete
SUBSET=full

bigcodebench.evaluate \
  --model $MODEL \
  --split $SPLIT \
  --subset $SUBSET \
  --execution "local" \
  --backend $BACKEND \
  --temperature 1.0 \
  --max_new_tokens 20000 \
  --pass_k "1,5,10"

# bigcodebench.inspect \
#   --eval_results bcb_results/Qwen--Qwen3-14B--main--bigcodebench-complete--vllm-1.0-1-sanitized_calibrated_eval_results.json \
#   --split complete
