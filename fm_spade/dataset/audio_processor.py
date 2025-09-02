# pylint: disable=C0301
'''
This module contains the AudioProcessor class and related functions for processing audio data.
It utilizes various libraries and models to perform tasks such as preprocessing, feature extraction,
and audio separation. The class is initialized with configuration parameters and can process
audio files using the provided models.
'''
import math
import os

import librosa
import numpy as np
import torch
from audio_separator.separator import Separator
from einops import rearrange
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import BaseModelOutput
import argparse
import torch.nn.functional as F

class Wav2VecModel(Wav2Vec2Model):
    """
    Wav2VecModel is a custom model class that extends the Wav2Vec2Model class from the transformers library. 
    It inherits all the functionality of the Wav2Vec2Model and adds additional methods for feature extraction and encoding.
    ...

    Attributes:
        base_model (Wav2Vec2Model): The base Wav2Vec2Model object.

    Methods:
        forward(input_values, seq_len, attention_mask=None, mask_time_indices=None
        , output_attentions=None, output_hidden_states=None, return_dict=None):
            Forward pass of the Wav2VecModel. 
            It takes input_values, seq_len, and other optional parameters as input and returns the output of the base model.

        feature_extract(input_values, seq_len):
            Extracts features from the input_values using the base model.

        encode(extract_features, attention_mask=None, mask_time_indices=None, output_attentions=None, output_hidden_states=None, return_dict=None):
            Encodes the extracted features using the base model and returns the encoded features.
    """
    def forward(
        self,
        input_values,
        seq_len,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Forward pass of the Wav2Vec model.

        Args:
            self: The instance of the model.
            input_values: The input values (waveform) to the model.
            seq_len: The sequence length of the input values.
            attention_mask: Attention mask to be used for the model.
            mask_time_indices: Mask indices to be used for the model.
            output_attentions: If set to True, returns attentions.
            output_hidden_states: If set to True, returns hidden states.
            return_dict: If set to True, returns a BaseModelOutput instead of a tuple.

        Returns:
            The output of the Wav2Vec model.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    def feature_extract(
        self,
        input_values,
        seq_len,
    ):
        """
        Extracts features from the input values and returns the extracted features.

        Parameters:
        input_values (torch.Tensor): The input values to be processed.
        seq_len (torch.Tensor): The sequence lengths of the input values.

        Returns:
        extracted_features (torch.Tensor): The extracted features from the input values.
        """
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        extract_features = linear_interpolation(extract_features, seq_len=seq_len)

        return extract_features

    def encode(
        self,
        extract_features,
        attention_mask=None,
        mask_time_indices=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Encodes the input features into the output space.

        Args:
            extract_features (torch.Tensor): The extracted features from the audio signal.
            attention_mask (torch.Tensor, optional): Attention mask to be used for padding.
            mask_time_indices (torch.Tensor, optional): Masked indices for the time dimension.
            output_attentions (bool, optional): If set to True, returns the attention weights.
            output_hidden_states (bool, optional): If set to True, returns all hidden states.
            return_dict (bool, optional): If set to True, returns a BaseModelOutput instead of the tuple.

        Returns:
            The encoded output features.
        """
        self.config.output_attentions = True

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, ) + encoder_outputs[1:]
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def linear_interpolation(features, seq_len):
    """
    Transpose the features to interpolate linearly.

    Args:
        features (torch.Tensor): The extracted features to be interpolated.
        seq_len (torch.Tensor): The sequence lengths of the features.

    Returns:
        torch.Tensor: The interpolated features.
    """
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class AudioProcessor:
    """
    AudioProcessor is a class that handles the processing of audio files.
    It takes care of preprocessing the audio files, extracting features
    using wav2vec models, and separating audio signals if needed.

    :param sample_rate: Sampling rate of the audio file
    :param fps: Frames per second for the extracted features
    :param wav2vec_model_path: Path to the wav2vec model
    :param only_last_features: Whether to only use the last features
    :param audio_separator_model_path: Path to the audio separator model
    :param audio_separator_model_name: Name of the audio separator model
    :param cache_dir: Directory to cache the intermediate results
    :param device: Device to run the processing on
    """
    def __init__(
        self,
        sample_rate,
        fps,
        wav2vec_model_path,
        only_last_features,
        audio_separator_model_path:str=None,
        audio_separator_model_name:str=None,
        cache_dir:str='',
        device="cpu",
    ) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        self.device = device

        self.audio_encoder = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).to(device=device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.only_last_features = only_last_features

        if audio_separator_model_name is not None:
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except OSError as _:
                print("Fail to create the output cache dir.")
            self.audio_separator = Separator(
                output_dir=cache_dir,
                output_single_stem="vocals",
                model_file_dir=audio_separator_model_path,
            )
            self.audio_separator.load_model(audio_separator_model_name)
            assert self.audio_separator.model_instance is not None, "Fail to load audio separate model."
        else:
            self.audio_separator=None
            print("Use audio directly without vocals seperator.")


        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path, local_files_only=True)


    def get_embedding(self, wav_file: str, target_len: int):
        """
        Preprocess wav audio file and convert to embeddings aligned with motion length.
    
        Args:
            wav_file (str): Path to the WAV file.
            target_len (int): Target length for alignment (motion length).
    
        Returns:
            torch.tensor: Audio embedding aligned to target length.
        """
        # 1. 加载音频
        speech_array, sampling_rate = librosa.load(wav_file, sr=self.sample_rate)
        assert sampling_rate == self.sample_rate, f"The audio sample rate must be {self.sample_rate}"
    
        # 2. 提取特征输入 (原始 waveform -> wav2vec2 输入)
        input_tensor = np.squeeze(self.wav2vec_feature_extractor(speech_array, sampling_rate=sampling_rate).input_values)
        input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).to(self.device)
    
        # 3. 设置目标长度 (与 motion 对齐)
        seq_len = target_len
    
        # 4. 提取 audio embedding (插值到 target_len)
        with torch.no_grad():
            embeddings = self.audio_encoder(input_tensor, seq_len=seq_len, output_hidden_states=True)
    
        assert len(embeddings) > 0, "Fail to extract audio embedding"
    
        # 5. 提取最后或所有隐藏层特征
        if self.only_last_features:
            audio_emb = embeddings.last_hidden_state.squeeze(0)  # shape: [target_len, feat_dim]
        else:
            audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
            audio_emb = rearrange(audio_emb, "b s d -> s b d")
    
        # 6. 检查长度并对齐
        current_len = audio_emb.shape[0]
        if current_len > target_len:
            audio_emb = audio_emb[:target_len]  # 截断
        elif current_len < target_len:
            pad_len = target_len - current_len
            audio_emb = F.pad(audio_emb, (0, 0, 0, pad_len), value=0.0)  # 填充 0
    
        return audio_emb.cpu().detach()

    def close(self):
        """
        TODO: to be implemented
        """
        return self

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
