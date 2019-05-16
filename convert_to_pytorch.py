import torch
import numpy as np
import tensorflow as tf

from vggish import VGGish
from audioset import vggish_params, vggish_slim, vggish_input


""" 
Script which converts the pretrained TensorFlow implementation of VGGish to a PyTorch equivalent, along with
a basic smoke test to verify accuracy.
"""


def main():
    with tf.Graph().as_default(), tf.Session() as sess:
        # -------------------
        # Step 1
        # -------------------
        # Load the model.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt')

        # Get all of the variables, and use this to construct a dictionary which maps
        # the name of the variables to their values.
        variables = tf.all_variables()
        variables = [x.name for x in variables]
        variable_values = sess.run(variables)
        variable_dict = dict(zip(variables, variable_values))

        # Create a new state dictionary which maps the TensorFlow version of the weights
        # to those in in the new PyTorch model.
        pytorch_model = VGGish()
        pytorch_feature_dict = pytorch_model.features.state_dict()
        pytorch_fc_dict = pytorch_model.fc.state_dict()

        # -------------------
        # Step 2
        # -------------------
        # There is a bias and weight vector for each convolution layer. The weights are not necessarily stored
        # in the same format and order between the two frameworks; for the TensorFlow model, the 12 vectors for the
        # convolution layers are first, followed by the 6 FC layers.
        tf_feature_names = list(variable_dict.keys())[:-6]
        tf_fc_names = list(variable_dict.keys())[-6:]

        def to_pytorch_tensor(weights):
            if len(weights.shape) == 4:
                tensor = torch.from_numpy(weights.transpose(3, 2, 0, 1)).float()
            else:
                tensor = torch.from_numpy(weights.T).float()
            return tensor

        # Convert the weights for the convolution layers.
        for tf_name, pytorch_name in zip(tf_feature_names, pytorch_feature_dict.keys()):
            print(f'Converting [{tf_name}] ---------->  [feature.{pytorch_name}]')
            pytorch_feature_dict[pytorch_name] = to_pytorch_tensor(variable_dict[tf_name])

        # Convert the weights for the FC layers.
        for tf_name, pytorch_name in zip(tf_fc_names, pytorch_fc_dict.keys()):
            print(f'Converting [{tf_name}] ---------->  [fc.{pytorch_name}]')
            pytorch_fc_dict[pytorch_name] = to_pytorch_tensor(variable_dict[tf_name])

        # -------------------
        # Step 3
        # -------------------
        # Load the new state dictionaries into the PyTorch model.
        pytorch_model.features.load_state_dict(pytorch_feature_dict)
        pytorch_model.fc.load_state_dict(pytorch_fc_dict)

        # -------------------
        # Step 4
        # -------------------
        # Generate a sample input (as in the AudioSet repo smoke test).
        num_secs = 3
        freq = 1000
        sr = 44100
        t = np.linspace(0, num_secs, int(num_secs * sr))
        x = np.sin(2 * np.pi * freq * t)

        # Produce a batch of log mel spectrogram examples.
        input_batch = vggish_input.waveform_to_examples(x, sr)

        # Run inference on the TensorFlow model.
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
        [tf_output] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: input_batch})

        # Run on the PyTorch model.
        pytorch_model = pytorch_model.to('cpu')
        pytorch_output = pytorch_model(torch.from_numpy(input_batch).unsqueeze(dim=1).float())
        pytorch_output = pytorch_output.detach().numpy()

        # -------------------
        # Step 5
        # -------------------
        # Compare the difference between the outputs.
        diff = np.linalg.norm(pytorch_output - tf_output) ** 2
        print(f'Distance between TensorFlow and PyTorch outputs: [{diff}]')
        assert diff < 1e-6

        # Run a smoke test.
        expected_embedding_mean = 0.131
        expected_embedding_std = 0.238

        # Verify the TF output.
        np.testing.assert_allclose(
            [np.mean(tf_output), np.std(tf_output)],
            [expected_embedding_mean, expected_embedding_std],
            rtol=0.001)

        # Verify the PyTorch output.
        np.testing.assert_allclose(
            [np.mean(pytorch_output), np.std(pytorch_output)],
            [expected_embedding_mean, expected_embedding_std],
            rtol=0.001)

        # -------------------
        # Step 6
        # -------------------
        print('Smoke test passed! Saving PyTorch weights to "pytorch_vggish.pth".')
        torch.save(pytorch_model.state_dict(), 'pytorch_vggish.pth')


if __name__ == '__main__':
    main()
