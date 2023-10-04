import datasets
from datasets import load_dataset

from wav2vec2 import CustomModelWav2Vec2

if __name__ == '__main__':
    model = CustomModelWav2Vec2()

    dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train", streaming=True)
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    dataset_iter = iter(dataset)
    sample = next(dataset_iter)
    x = sample["audio"]["array"]
    outputs = model(x)
    print(outputs)
