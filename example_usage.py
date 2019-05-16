import torch
import numpy as np

from vggish import VGGish
from audioset import vggish_input, vggish_postprocess


def main():
    # Initialize the PyTorch model.
    device = 'cuda:0'
    pytorch_model = VGGish()
    pytorch_model.load_state_dict(torch.load('pytorch_vggish.pth'))
    pytorch_model = pytorch_model.to(device)

    # Generate a sample input (as in the AudioSet repo smoke test).
    num_secs = 3
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)

    # Produce a batch of log mel spectrogram examples.
    input_batch = vggish_input.waveform_to_examples(x, sr)
    input_batch = torch.from_numpy(input_batch).unsqueeze(dim=1)
    input_batch = input_batch.float().to(device)

    # Run the PyTorch model.
    pytorch_output = pytorch_model(input_batch)
    pytorch_output = pytorch_output.detach().cpu().numpy()
    print('Input Shape:', tuple(input_batch.shape))
    print('Output Shape:', tuple(pytorch_output.shape))

    expected_embedding_mean = 0.131
    expected_embedding_std = 0.238
    print('Computed Embedding Mean and Standard Deviation:', np.mean(pytorch_output), np.std(pytorch_output))
    print('Expected Embedding Mean and Standard Deviation:', expected_embedding_mean, expected_embedding_std)

    # Post-processing.
    post_processor = vggish_postprocess.Postprocessor('vggish_pca_params.npz')
    postprocessed_output = post_processor.postprocess(pytorch_output)
    expected_postprocessed_mean = 123.0
    expected_postprocessed_std = 75.0
    print('Computed Post-processed Embedding Mean and Standard Deviation:', np.mean(postprocessed_output),
          np.std(postprocessed_output))
    print('Expected Post-processed Embedding Mean and Standard Deviation:', expected_postprocessed_mean,
          expected_postprocessed_std)


if __name__ == '__main__':
    main()
