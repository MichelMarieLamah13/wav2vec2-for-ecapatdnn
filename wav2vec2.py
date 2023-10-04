import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def get_output_rep(hidden_states, state_dict):
    learnable_weights = []
    learnable_bias = []

    # Learnable weights and bias of the cnn encoder
    ln_weight = state_dict['wav2vec2.feature_projection.layer_norm.weight']
    ln_bias = state_dict['wav2vec2.feature_projection.layer_norm.bias']

    p_weight = state_dict['wav2vec2.feature_projection.projection.weight']
    p_bias = state_dict['wav2vec2.feature_projection.projection.bias']

    # we project the weights and bias to match the dim of the encoder
    ln_weight = ln_weight @ p_weight + p_bias
    ln_bias = ln_bias @ p_weight + p_bias

    learnable_weights.append(ln_weight)
    learnable_bias.append(ln_bias)

    nb_layers = len(hidden_states)
    for i in range(nb_layers - 1):
        # Learnable weights and bias of the  transformer i
        learnable_weights.append(state_dict[f'wav2vec2.encoder.layers.{i}.final_layer_norm.weight'])
        learnable_bias.append(state_dict[f'wav2vec2.encoder.layers.{i}.final_layer_norm.bias'])

    result = []
    nb_frames = hidden_states[0].shape[1]

    for t in range(nb_frames):

        for l in range(nb_layers):
            o_t = learnable_weights[l] @ hidden_states[0, l, :]

    return result


class CustomModelWav2Vec2(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, output_hidden_states=True)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def forward(self, x):
        x = self.processor(x, return_tensor='pt', padding="longest", sampling_rate=16_000)
        x = torch.Tensor(x.input_values)

        with torch.no_grad():
            output = self.model(x)

        hidden_states = list(output.hidden_states)
        state_dict = self.model.state_dict()

        get_output_rep(hidden_states, state_dict)

        return get_output_rep(hidden_states, state_dict)
