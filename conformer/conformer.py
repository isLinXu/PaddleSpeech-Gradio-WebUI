import paddle
import soundfile

import warnings
warnings.filterwarnings('ignore')

from yacs.config import CfgNode
from paddlespeech.audio.transform.spectrogram import LogMelSpectrogramKaldi
from paddlespeech.audio.transform.cmvn import GlobalCMVN
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.models.u2 import U2Model

from matplotlib import pyplot as plt

checkpoint_path = "work/workspace_asr/exp/transformer/checkpoints/avg_20.pdparams"
audio_file = "work/workspace_asr/data/demo_01_03.wav"



# 读取 conf 文件并结构化
transformer_config = CfgNode(new_allowed=True)
transformer_config.merge_from_file("work/workspace_asr/conf/transformer.yaml")
transformer_config.decoding.decoding_method = "attention"
# print(transformer_config)

transformer_config.collator.vocab_filepath = "work/workspace_asr/data/lang_char/vocab.txt"
transformer_config.collator.augmentation_config = "work/workspace_asr/conf/preprocess.yaml"
# 构建 logmel 特征
logmel_kaldi= LogMelSpectrogramKaldi(
            fs= 16000,
            n_mels= 80,
            n_shift= 160,
            win_length= 400,
            dither= True)

# 特征减均值除以方差
cmvn = GlobalCMVN(
    cmvn_path="work/workspace_asr/data/mean_std.json"
)

array, _ = soundfile.read(audio_file, dtype="int16")

array = logmel_kaldi(array, train=False)

audio_feature_i = cmvn(array)

audio_len = audio_feature_i.shape[0]
audio_len = paddle.to_tensor(audio_len)

audio_feature = paddle.to_tensor(audio_feature_i, dtype='float32')
# (B, T, D)
audio_feature = paddle.unsqueeze(audio_feature, axis=0)
# print (audio_feature.shape)

plt.figure()
plt.imshow(audio_feature_i.T, origin='lower')
plt.show()



# 模型配置
model_conf = transformer_config.model
# input_dim 存储的是特征的纬度
model_conf.input_dim = 80
# output_dim 存储的字表的长度
model_conf.output_dim = 4233 
print ("model_conf", model_conf)

model = U2Model.from_config(model_conf)

# 加载预训练的模型
model_dict = paddle.load(checkpoint_path)
model.set_state_dict(model_dict)


# 预测
decoding_config = transformer_config.decoding
text_feature = TextFeaturizer(unit_type='char',
                            vocab=transformer_config.collator.vocab_filepath)


result_transcripts = model.decode(
            audio_feature,
            audio_len,
            text_feature=text_feature,
            decoding_method=decoding_config.decoding_method,
            beam_size=decoding_config.beam_size,
            ctc_weight=decoding_config.ctc_weight,
            decoding_chunk_size=decoding_config.decoding_chunk_size,
            num_decoding_left_chunks=decoding_config.num_decoding_left_chunks,
            simulate_streaming=decoding_config.simulate_streaming)

print(result_transcripts)

print ("预测结果对应的token id为:")
print (result_transcripts[1][0])
print ("预测结果为:")
print (result_transcripts[0][0])