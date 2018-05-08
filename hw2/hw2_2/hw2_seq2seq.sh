 #!/bin/bash
wget -O 'lstm_model_attention.ckpt.data-00000-of-00001' 'https://www.dropbox.com/s/c10ov31a3y8jrq2/lstm_model_attention.ckpt.data-00000-of-00001?dl=1'
wget -O 'lstm_model_attention.ckpt.index' 'https://www.dropbox.com/s/lixsed7v570yna7/lstm_model_attention.ckpt.index?dl=1'
wget -O 'lstm_model_attention.ckpt.meta' 'https://www.dropbox.com/s/q68xj8hivp2kx9q/lstm_model_attention.ckpt.meta?dl=1'
python3 inference.py $1 $2
