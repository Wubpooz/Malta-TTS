# Logs Training
```
/content/Malta-TTS/FineTuning/NewLanguage
Finetuning for mt
Step 1: Downloading XTTS base model files.
 > Downloading XTTS v-main files...
 > XTTS model files downloaded successfully!
Step 2: Extending the XTTS tokenizer with the new language.
No new tokens to add by heuristic.
Tokenizer saved to /content/drive/MyDrive/XTTS_Maltese_Training/output/tokenizer.json and copied to vocab.json
Backing up checkpoint: /content/drive/MyDrive/XTTS_Maltese_Training/output/model.pth -> model_backup.pth
Resizing checkpoint embeddings: /content/drive/MyDrive/XTTS_Maltese_Training/output/model.pth
/content/Malta-TTS/FineTuning/NewLanguage/tokenizer_extension.py:52: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  checkpoint = torch.load(xtts_checkpoint_path, map_location="cpu")
Current vocab size: 10368, New vocab size: 10368
Vocabulary sizes match, no resizing needed.
Updated config file saved to /content/drive/MyDrive/XTTS_Maltese_Training/output/config.json. Added new language: mt. Vocab size: 10368
Extended vocabulary size: 10368
Config vocab size: 10368
Step 3: Starting GPT training.
Using updated checkpoint: /content/drive/MyDrive/XTTS_Maltese_Training/output/model.pth
Using updated tokenizer: /content/drive/MyDrive/XTTS_Maltese_Training/output/vocab.json
Using vocab size: 10368
 > Training XTTS model for Maltese with 1 datasets, 1 epochs, batch size 1, grad_acumm 48, output path: /content/drive/MyDrive/XTTS_Maltese_Training/output/training
 > Using the following datasets:
/content/drive/MyDrive/XTTS_Maltese_Data_20KHz/metadata_train.csv /content/drive/MyDrive/XTTS_Maltese_Data_20KHz/metadata_eval.csv mt
Setting up model arguments...
/usr/local/lib/python3.11/dist-packages/TTS/utils/io.py:54: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  return torch.load(f, map_location=map_location, **kwargs)
/usr/local/lib/python3.11/dist-packages/TTS/tts/layers/tortoise/arch_utils.py:336: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  self.mel_norms = torch.load(f)
/usr/local/lib/python3.11/dist-packages/TTS/tts/layers/xtts/trainer/gpt_trainer.py:185: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  dvae_checkpoint = torch.load(self.args.dvae_checkpoint, map_location=torch.device("cpu"))
>> DVAE weights restored from: /content/drive/MyDrive/XTTS_Maltese_Training/output/dvae.pth
Loading datasets...
 | > Found 3984 files in /content/drive/MyDrive/XTTS_Maltese_Data_20KHz
 > Loaded 3984 training samples and 997 evaluation samples.
 > Training Environment:
 | > Backend: Torch
 | > Mixed precision: False
 | > Precision: float32
 | > Current device: 0
 | > Num. of GPUs: 1
 | > Num. of CPUs: 2
 | > Num. of Torch Threads: 1
 | > Torch seed: 1
 | > Torch CUDNN: True
 | > Torch CUDNN deterministic: False
 | > Torch CUDNN benchmark: False
 | > Torch TF32 MatMul: False
2025-08-19 14:22:45.265939: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:467] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1755613365.539258    9417 cuda_dnn.cc:8579] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1755613365.614624    9417 cuda_blas.cc:1407] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
W0000 00:00:1755613366.163048    9417 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1755613366.163092    9417 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1755613366.163097    9417 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
W0000 00:00:1755613366.163102    9417 computation_placer.cc:177] computation placer already registered. Please check linkage and avoid linking the same target more than once.
 > Start Tensorboard: tensorboard --logdir=/content/drive/MyDrive/XTTS_Maltese_Training/output/training/GPT_XTTS_FT-August-19-2025_02+22PM-2ae7a3e

 > Model has 525996710 parameters
Adding new language: mt
New language added: mt
Starting training...

 > EPOCH: 0/1
 --> /content/drive/MyDrive/XTTS_Maltese_Training/output/training/GPT_XTTS_FT-August-19-2025_02+22PM-2ae7a3e
 > Sampling by language: dict_keys(['mt'])

 > TRAINING (2025-08-19 14:22:52) 

   --> TIME: 2025-08-19 14:22:56 -- STEP: 0/3984 -- GLOBAL_STEP: 0
     | > loss_text_ce: 0.22755847871303558  (0.22755847871303558)
     | > loss_mel_ce: 5.383345127105713  (5.383345127105713)
     | > loss: 0.11689382791519165  (0.11689382791519165)
     | > current_lr: 5e-06 
     | > step_time: 0.8086  (0.8086225986480713)
     | > loader_time: 3.3665  (3.3664534091949463)


   --> TIME: 2025-08-19 14:23:42 -- STEP: 200/3984 -- GLOBAL_STEP: 200
     | > loss_text_ce: 0.21218353509902954  (0.24113701090216635)
     | > loss_mel_ce: 4.792850971221924  (nan)
     | > loss: 0.1042715534567833  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1687  (0.1479287660121917)
     | > loader_time: 0.0095  (0.015414910316467278)


   --> TIME: 2025-08-19 14:24:28 -- STEP: 400/3984 -- GLOBAL_STEP: 400
     | > loss_text_ce: 0.20771436393260956  (0.2372326312586665)
     | > loss_mel_ce: 4.763483047485352  (nan)
     | > loss: 0.10356661677360535  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1162  (0.14828080296516422)
     | > loader_time: 0.0091  (0.014332452416419983)


   --> TIME: 2025-08-19 14:25:12 -- STEP: 600/3984 -- GLOBAL_STEP: 600
     | > loss_text_ce: 0.22492223978042603  (0.23759328377743563)
     | > loss_mel_ce: 4.4374470710754395  (nan)
     | > loss: 0.09713269770145416  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.121  (0.14757777134577438)
     | > loader_time: 0.0078  (0.013066962957382194)


   --> TIME: 2025-08-19 14:25:58 -- STEP: 800/3984 -- GLOBAL_STEP: 800
     | > loss_text_ce: 0.19285964965820312  (0.23543215092271566)
     | > loss_mel_ce: 5.090768814086914  (nan)
     | > loss: 0.11007559299468994  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1625  (0.1479025691747667)
     | > loader_time: 0.0084  (0.012694222629070278)


   --> TIME: 2025-08-19 14:26:45 -- STEP: 1000/3984 -- GLOBAL_STEP: 1000
     | > loss_text_ce: 0.2893211543560028  (0.23088019873201848)
     | > loss_mel_ce: 4.799831867218018  (nan)
     | > loss: 0.10602401942014694  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.117  (0.14913334941864034)
     | > loader_time: 0.0095  (0.012335450410842892)


   --> TIME: 2025-08-19 14:27:34 -- STEP: 1200/3984 -- GLOBAL_STEP: 1200
     | > loss_text_ce: 0.27572888135910034  (0.22833806475003557)
     | > loss_mel_ce: 4.361039161682129  (nan)
     | > loss: 0.09659933298826218  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1077  (0.15063963909943923)
     | > loader_time: 0.0111  (0.01205707053343454)


   --> TIME: 2025-08-19 14:28:21 -- STEP: 1400/3984 -- GLOBAL_STEP: 1400
     | > loss_text_ce: 0.19423453509807587  (0.22591488101652688)
     | > loss_mel_ce: 4.447282791137695  (nan)
     | > loss: 0.09669827669858932  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1721  (0.15121245469365813)
     | > loader_time: 0.0148  (0.011787546702793666)


   --> TIME: 2025-08-19 14:29:09 -- STEP: 1600/3984 -- GLOBAL_STEP: 1600
     | > loss_text_ce: 0.19451627135276794  (0.22307939922437067)
     | > loss_mel_ce: 4.391462802886963  (nan)
     | > loss: 0.09554123133420944  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1121  (0.1517107413709162)
     | > loader_time: 0.0072  (0.011751018166542053)


   --> TIME: 2025-08-19 14:29:56 -- STEP: 1800/3984 -- GLOBAL_STEP: 1800
     | > loss_text_ce: 0.22472961246967316  (0.22102927808132433)
     | > loss_mel_ce: 4.030942916870117  (nan)
     | > loss: 0.0886598452925682  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1773  (0.15197158561812488)
     | > loader_time: 0.0116  (0.011520591974258422)


   --> TIME: 2025-08-19 14:30:44 -- STEP: 2000/3984 -- GLOBAL_STEP: 2000
     | > loss_text_ce: 0.1710306704044342  (0.21878219324350354)
     | > loss_mel_ce: 4.442059516906738  (nan)
     | > loss: 0.09610604494810104  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1367  (0.15238851749896987)
     | > loader_time: 0.0082  (0.011438338994979858)


   --> TIME: 2025-08-19 14:31:32 -- STEP: 2200/3984 -- GLOBAL_STEP: 2200
     | > loss_text_ce: 0.16353975236415863  (0.2167401518130844)
     | > loss_mel_ce: 4.8388471603393555  (nan)
     | > loss: 0.10421639680862427  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1292  (0.1529119697484101)
     | > loader_time: 0.0088  (0.011319132609800862)


   --> TIME: 2025-08-19 14:32:21 -- STEP: 2400/3984 -- GLOBAL_STEP: 2400
     | > loss_text_ce: 0.2828022241592407  (0.2145825321413577)
     | > loss_mel_ce: 3.2861554622650146  (nan)
     | > loss: 0.07435329258441925  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1529  (0.1532701882719992)
     | > loader_time: 0.0109  (0.01124754508336386)


   --> TIME: 2025-08-19 14:33:07 -- STEP: 2600/3984 -- GLOBAL_STEP: 2600
     | > loss_text_ce: 0.1360827535390854  (0.21262868242195024)
     | > loss_mel_ce: 4.577248573303223  (nan)
     | > loss: 0.09819440543651581  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1396  (0.15317315743519694)
     | > loader_time: 0.008  (0.011188767965023343)


   --> TIME: 2025-08-19 14:33:53 -- STEP: 2800/3984 -- GLOBAL_STEP: 2800
     | > loss_text_ce: 0.18691657483577728  (0.21076072590159523)
     | > loss_mel_ce: 3.9437692165374756  (nan)
     | > loss: 0.08605595678091049  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1345  (0.15291924170085353)
     | > loader_time: 0.0095  (0.011105595571654192)


   --> TIME: 2025-08-19 14:34:41 -- STEP: 3000/3984 -- GLOBAL_STEP: 3000
     | > loss_text_ce: 0.14514359831809998  (0.2085119023273389)
     | > loss_mel_ce: 4.687170505523682  (nan)
     | > loss: 0.10067321360111237  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1595  (0.1531027445793152)
     | > loader_time: 0.0086  (0.01111129625638326)


   --> TIME: 2025-08-19 14:35:29 -- STEP: 3200/3984 -- GLOBAL_STEP: 3200
     | > loss_text_ce: 0.18211399018764496  (0.20650915422942492)
     | > loss_mel_ce: 3.954969644546509  (nan)
     | > loss: 0.08618924021720886  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1625  (0.15313900262117353)
     | > loader_time: 0.0241  (0.011049595102667809)


   --> TIME: 2025-08-19 14:36:16 -- STEP: 3400/3984 -- GLOBAL_STEP: 3400
     | > loss_text_ce: 0.16159699857234955  (0.20462788640137985)
     | > loss_mel_ce: 3.7073922157287598  (nan)
     | > loss: 0.0806039422750473  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.164  (0.15324610710144024)
     | > loader_time: 0.0083  (0.01104233931092655)


   --> TIME: 2025-08-19 14:37:04 -- STEP: 3600/3984 -- GLOBAL_STEP: 3600
     | > loss_text_ce: 0.14153452217578888  (0.2027974851098326)
     | > loss_mel_ce: 2.74570369720459  (nan)
     | > loss: 0.06015079841017723  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1591  (0.15338963554965107)
     | > loader_time: 0.0095  (0.01096440335114797)


   --> TIME: 2025-08-19 14:37:50 -- STEP: 3800/3984 -- GLOBAL_STEP: 3800
     | > loss_text_ce: 0.16177041828632355  (0.2010479261275185)
     | > loss_mel_ce: 3.44710636138916  (nan)
     | > loss: 0.07518493384122849  (nan)
     | > current_lr: 5e-06 
     | > step_time: 0.1044  (0.15321841032881467)
     | > loader_time: 0.0074  (0.010925197852285285)

 > Filtering invalid eval samples!!
 > Total eval samples after filtering: 996

 > EVALUATION 


  --> EVAL PERFORMANCE
     | > avg_loader_time: 0.039930398260528646 (+0)
     | > avg_loss_text_ce: 0.16418918056703685 (+0)
     | > avg_loss_mel_ce: 4.1349405672082895 (+0)
     | > avg_loss: 4.299129742953043 (+0)

 > BEST MODEL : /content/drive/MyDrive/XTTS_Maltese_Training/output/training/GPT_XTTS_FT-August-19-2025_02+22PM-2ae7a3e/best_model_3984.pth
Training finished!
Saving final model...

 > CHECKPOINT : /content/drive/MyDrive/XTTS_Maltese_Training/output/training/GPT_XTTS_FT-August-19-2025_02+22PM-2ae7a3e/checkpoint_3984.pth
Saving configuration...
Configuration saved to /content/drive/MyDrive/XTTS_Maltese_Training/output/training/config.json and model checkpoint saved to /content/drive/MyDrive/XTTS_Maltese_Training/output/training/final_model.pth.
Speaker reference: /content/drive/MyDrive/XTTS_Maltese_Data_20KHz/wavs/MSRHS_M_11_P24U082_0147.wav
Checkpoint saved in dir: /content/drive/MyDrive/XTTS_Maltese_Training/output/training/GPT_XTTS_FT-August-19-2025_02+22PM-2ae7a3e
Finetuning process completed!
```