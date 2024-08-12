# export MODEL_DIR=/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/DataAugmentation/prism/m39v1/


import os
from prism import Prism
MODEL_DIR = "/Users/diptisengupta/Desktop/CODEWORK/GitHub/WS2022/Pretrained-Language-Model/HPO/ray_cluster_test/BoHBCode/DataAugmentation/prism/m39v1"
os.environ['MODEL_DIR'] = MODEL_DIR



if __name__ == '__main__':
    prism = Prism(model_dir=os.environ['MODEL_DIR'], lang='de')
    sentences = [
    "Dies ist ein Testsatz.",
    "Prism ist ein leistungsstarkes Werkzeug zur maschinellen Übersetzungsbewertung."
    ]
    for sentence in sentences:
        paraphrased_sentence = prism.generate(sentence, beam=5)
        print(f"Original: {sentence}")
        print(f"Paraphrase: {paraphrased_sentence}")