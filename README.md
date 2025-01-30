# Bert-arabic-fake-news-rbf
## Overview
Bert-arabic-fake-news-rbf is a project aimed at detecting fake news in Arabic using a Recurrent Basis Function (RBF) network combined with BERT embeddings. This project leverages the power of BERT for natural language understanding and the RBF network for classification tasks.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Dataset
The dataset used for this project consists of Arabic news articles labeled as real or fake. The dataset is preprocessed to be compatible with BERT embeddings.

## Model Architecture
The model architecture consists of:
1. **BERT Embeddings**: To convert text into high-dimensional vectors.
2. **RBF Network**: To classify the embeddings into real or fake news.

## Training
To train the model, run:
```python
python train.py --EPOCHS 5 --device mps --BATCH_SIZE 8 --enable_rbf 1 --num_kernels 1 --kernel_name gaussian
```

## Results
The model achieves an accuracy of X% on the test set. Detailed results and performance metrics can be found in the `results` directory.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
