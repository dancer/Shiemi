# shiemi

A character-based language model trained on Blue Exorcist data.

<img src="anime.gif" alt="Anime Demo" width="500"/>

▲ Coming Soon: Support for all anime series, not just Blue Exorcist!  
► Website (Coming Soon): https://shiemi.com  
► For sponsorship inquiries: josh@afterima.ge

#### ▫ Overview

Shiemi is a transformer-based language model designed to generate text in the style of Blue Exorcist characters. Built with PyTorch and trained on dialogue and descriptions from the series.

#### ▪ Features

▫ Transformer architecture with:
└─ 8 layers
└─ 8 attention heads
└─ 512 model dimensions
└─ 4000 vocabulary size

▫ Character-based generation  
▫ Interactive chat interface  
▫ Efficient tokenization with SentencePiece  
▫ Mixed precision training support

#### ▫ Installation
```bash
# Clone the repository
git clone https://github.com/dancer/shiemi.git
cd shiemi

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### ▪ Usage

##### Training the Tokenizer
```bash
python -m shiemi train-tokenizer \
    --input_files data/your_data.txt \
    --output_dir tokenizer \
    --vocab_size 4000
```

##### Training the Model
```bash
python -m shiemi train \
    --train_files data/your_data.txt \
    --tokenizer_path tokenizer/shiemi.model \
    --output_dir model_output \
    --batch_size 8 \
    --num_epochs 50
```

##### Chat Interface
```bash
python -m shiemi chat \
    --model_path model_output/checkpoint_final.pt \
    --tokenizer_path tokenizer/shiemi.model \
    --max_length 200 \
    --temperature 0.7 \
    --top_p 0.9
```

#### ▫ Model Architecture
```
Input Text → Tokenizer → Transformer (8 Layers, 8 Heads, 512d) → Generated Text
```

#### ▪ Configuration
Model parameters can be adjusted in `shiemi/config/model_config.py`:
```python
n_layers = 8
n_heads = 8
d_model = 512
d_ff = 2048
vocab_size = 4000
max_seq_length = 512
```

#### ▫ License
This project is licensed under the MIT License - see the LICENSE file for details.

#### ▪ Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

#### ▫ Acknowledgments
- Built with PyTorch
- Tokenization by SentencePiece
- Inspired by Blue Exorcist